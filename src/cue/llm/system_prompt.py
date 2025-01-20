import platform

SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an machine using {platform.machine()} architecture with internet access.
* When using bash tool, where possible/feasible, try to chain multiple of these calls all into one function calls request.
</SYSTEM_CAPABILITY>
"""  # noqa: E501
