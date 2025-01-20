import os
import json
import asyncio
import logging
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class ServerContext:
    def __init__(self, client_ctx, session, session_instance):
        self.client_ctx = client_ctx
        self.session = session
        self.session_instance = session_instance


class MCPServerManager:
    def __init__(self, config_path: str = "mcp_config.json"):
        self.config_path = config_path
        self.servers: dict[str, dict[str, Any]] = {}
        self._initialized = False
        self._contexts: dict[str, ServerContext] = {}
        self.logger = logging.getLogger("MCPServerManager")
        self._task = None
        self._close_event = asyncio.Event()

    async def _init_server(self, server_name: str, server_params: StdioServerParameters):
        self.logger.debug(f"Initializing server: {server_name}")
        try:
            client_ctx = stdio_client(server_params)
            read, write = await client_ctx.__aenter__()

            session = ClientSession(read, write)
            session_instance = await session.__aenter__()
            await session_instance.initialize()

            self._contexts[server_name] = ServerContext(
                client_ctx=client_ctx, session=session, session_instance=session_instance
            )

            self.servers[server_name] = {"session": session_instance, "tools": await session_instance.list_tools()}
            self.logger.debug(f"Server {server_name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing server {server_name}: {e}")
            if server_name in self._contexts:
                await self._cleanup_server(server_name)
            raise

    async def _cleanup_server(self, server_name: str):
        if server_name not in self._contexts:
            return

        self.logger.debug(f"Cleaning up server: {server_name}")
        context = self._contexts[server_name]

        try:
            # Clean up in reverse order of initialization
            if context.session:
                self.logger.debug(f"Closing session for {server_name}")
                await context.session.__aexit__(None, None, None)

            if context.client_ctx:
                self.logger.debug(f"Closing client context for {server_name}")
                await context.client_ctx.__aexit__(None, None, None)

        except Exception as e:
            self.logger.error(f"Error during cleanup of {server_name}: {e}")
        finally:
            self._contexts.pop(server_name, None)
            self.servers.pop(server_name, None)
            self.logger.debug(f"Server {server_name} cleanup completed")

    async def connect(self):
        configs = self._load_config()
        if not configs:
            return
        self.logger.info("MCPServerManager connecting...")

        try:
            for server_name, server_params in configs.items():
                await self._init_server(server_name, server_params)
            self._initialized = True
            self.logger.info("MCPServerManager connected successfully")
        except Exception as e:
            self.logger.error(f"Error during connect: {e}")
            await self.disconnect()
            raise

    async def run(self):
        """Run the manager until shutdown is requested"""
        try:
            await self._close_event.wait()
        except asyncio.CancelledError:
            self.logger.info("Manager run cancelled")
            raise
        finally:
            await self.disconnect()

    def request_shutdown(self):
        """Request the manager to shut down"""
        self._close_event.set()

    async def disconnect(self):
        self.logger.info("MCPServerManager disconnecting...")
        if not self._initialized:
            return

        try:
            # Create cleanup tasks
            cleanup_tasks = []
            for server_name in reversed(list(self._contexts.keys())):
                task = asyncio.create_task(self._cleanup_server(server_name))
                cleanup_tasks.append(task)

            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        finally:
            self._initialized = False
            self._contexts.clear()
            self.servers.clear()
            self.logger.info("MCPServerManager disconnected")

    def _load_config(self) -> dict[str, StdioServerParameters]:
        if not os.path.exists(self.config_path):
            self.logger.warning(f"MCP config file not found, skip mcp: {self.config_path}")
            return {}

        try:
            with open(self.config_path) as f:
                config_data = json.load(f)

            server_configs = {}
            for server_name, config in config_data.get("mcpServers", {}).items():
                server_configs[server_name] = StdioServerParameters(
                    command=config["command"], args=config["args"], env=config.get("env")
                )
            return server_configs
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse config file {self.config_path}: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading config file {self.config_path}: {e}")
            return {}

    def get_available_servers(self) -> list[str]:
        return list(self.servers.keys())

    def get_server_tools(self, server_name: str) -> Optional[Any]:
        if server_name not in self.servers:
            return None
        return self.servers[server_name]["tools"]

    def list_tools_json(self) -> dict[str, Any]:
        """
        Get a dictionary of all tools across all servers, removing server information.
        The tool name will be unique key even if it exists in multiple servers.
        If a tool exists in multiple servers, it will use the first occurrence.

        Returns:
            Dict[str, Any]: Dictionary mapping tool names to their info
                {
                    'tool_name1': tool_info1,
                    'tool_name2': tool_info2,
                    ...
                }
        """
        all_tools = []
        for server_info in self.servers.values():
            tools = server_info["tools"]
            if hasattr(tools, "model_dump"):
                server_tools = tools.model_dump().get("tools", {})
                updated_tools = [self.convert_tool_schema(tool) for tool in server_tools]
                all_tools.extend(updated_tools)
        return all_tools

    def convert_tool_schema(self, tool):
        """Convert a single tool schema by replacing inputSchema with input_schema"""
        if isinstance(tool, dict):
            transformed = {}
            for key, value in tool.items():
                if key == "inputSchema":
                    transformed["input_schema"] = value
                else:
                    transformed[key] = value
            return transformed
        return tool

    def find_tool(self, tool_name: str) -> Optional[tuple[str, Any]]:
        for server_name, server_info in self.servers.items():
            tools = server_info["tools"]
            if hasattr(tools, "model_dump"):
                tools_dict = tools.model_dump()
                # The tools are in a list, not a dict, so we need to search through them
                tool_list = tools_dict.get("tools", [])
                for tool in tool_list:
                    if tool.get("name") == tool_name:
                        return server_name, tool
        return None

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Optional[Any]:
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not found")

        session = self.servers[server_name]["session"]
        try:
            return await session.call_tool(tool_name, arguments=arguments)
        except Exception as e:
            print(f"Error calling tool {tool_name} on server {server_name}: {str(e)}")
            raise
