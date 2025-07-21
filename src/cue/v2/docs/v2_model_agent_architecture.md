# V2 Model-Agent Architecture Design

## 🎯 Core Principles

### 1. **Single Responsibility Separation**
- **ModelBase**: Pure model response generation (single-turn)
- **AgentRunner**: Conversation orchestration (multi-turn)
- **Clear boundary**: One layer handles AI responses, another handles conversation flow

### 2. **Provider Abstraction**
- Each provider implements `ModelBase` interface
- Provider-specific logic contained within model classes
- Zero coupling between different providers

### 3. **Semantic Clarity**
- `get_response()` - explicit single model response
- `stream_response()` - explicit streaming model response
- Method names express intent, not implementation

## 🏗️ Architecture Layers

```
┌─────────────────────────────────────────┐
│             AgentRunner                 │  ← Multi-turn orchestration
│  - Conversation state management        │
│  - Turn coordination                    │
│  - Memory handling                      │
└─────────────────┬───────────────────────┘
                  │ delegates to
┌─────────────────▼───────────────────────┐
│             ModelBase                   │  ← Single-turn responses
│  - get_response(agent, inputs)          │
│  - stream_response(agent, inputs)       │
└─────────────────┬───────────────────────┘
                  │ implemented by
        ┌─────────┼─────────┬─────────────┐
        │         │         │             │
   ┌────▼────┐ ┌──▼──┐ ┌────▼────┐ ┌─────▼─────┐
   │Anthropic│ │OpenAI│ │ Gemini  │ │  Future   │
   │ Model   │ │Model │ │ Model   │ │ Providers │
   └─────────┘ └─────┘ └─────────┘ └───────────┘
```

## 🔧 Interface Design

### ModelBase Contract
```python
class ModelBase(ABC):
    async def get_response(agent, inputs) -> RunResult
    async def stream_response(agent, inputs, hooks) -> AsyncGenerator[StreamEvent]
```

**Responsibilities:**
- Single model interaction
- Provider-specific message formatting
- Tool execution within single turn
- Streaming event generation

### AgentRunner Contract
```python
class AgentRunner:
    def __init__(model: ModelBase)
    async def get_response(agent, inputs) -> RunResult
    async def stream_response(agent, inputs, hooks) -> AsyncGenerator[StreamEvent]
```

**Responsibilities:**
- Multi-turn conversation flow
- Agent state management
- Turn coordination
- Delegates to ModelBase for responses

## 🎭 Benefits

### **1. Clarity**
- **What**: ModelBase = model responses, AgentRunner = conversations
- **Why**: Clear mental model for developers
- **How**: Semantic method names express intent

### **2. Extensibility**
- **Models**: Add new providers by implementing ModelBase
- **Agents**: Enhance conversation logic in AgentRunner
- **Tools**: Provider-specific tool handling in models

### **3. Testability**
- **Unit**: Test model responses in isolation
- **Integration**: Test conversation flows separately
- **Mocking**: Easy to mock model responses for agent tests

### **4. Maintainability**
- **Provider changes**: Contained within model classes
- **Conversation logic**: Centralized in AgentRunner
- **Interface stability**: Abstract contracts protect against changes

## 🚀 Usage Patterns

### Direct Model Usage (Single-turn)
```python
model = get_model_for_name("claude-3-5-haiku")
result = await model.get_response(agent, inputs)
```

### Agent Usage (Multi-turn)
```python
model = get_model_for_name("claude-3-5-haiku")
agent_runner = AgentRunner(model)
result = await agent_runner.get_response(agent, inputs)
```

### Streaming with Hooks
```python
async for event in model.stream_response(agent, inputs, hooks):
    if event.type == "text":
        print(event.content, end="")
    elif event.is_final:
        final_result = event.accumulated
```

## 🔄 Evolution Path

### Current State
- **ModelBase**: Fully implemented with streaming compliance
- **AgentRunner**: Simple delegation to ModelBase
- **Providers**: All implement ModelBase interface

### Future Enhancements
- **AgentRunner**: Enhanced multi-turn logic, memory management
- **Models**: Advanced streaming features (thinking, server tools)
- **Tools**: Cross-turn tool state management

## 📐 Design Principles Applied

1. **Separation of Concerns**: Model vs Agent responsibilities
2. **Interface Segregation**: Focused, cohesive interfaces
3. **Dependency Inversion**: AgentRunner depends on ModelBase abstraction
4. **Single Responsibility**: Each class has one reason to change
5. **Open/Closed**: Open for extension (new providers), closed for modification

This architecture provides a clean, maintainable foundation for both simple model interactions and complex multi-turn conversations.