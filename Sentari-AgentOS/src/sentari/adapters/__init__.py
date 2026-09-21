"""Framework adapters: this is what "Sentari wraps your existing agent
framework instead of replacing it" means concretely. Each adapter routes a
specific framework's own tool-execution extension point through the real
Sentari kernel (quota-checked, optionally resource-mediated, audit-logged)
without reimplementing that framework's dispatch, state machine, or graph
semantics."""
