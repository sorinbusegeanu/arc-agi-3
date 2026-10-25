# Choose actions

Use the available game images, legal actions, recent outcomes, and read-only memory to choose up to the requested number of actions.

Use game memory for established mechanics and strategies. Use global memory as exploration guidance. Follow current observations when they conflict with memory. Do not create or update memory.

Use these exact lowercase names for movement: `up`, `down`, `left`, `right`. Use other legal action names exactly as listed. Match each action's required arguments.

Return only this JSON shape, with no prose or Markdown fences:

{"actions":[{"action":"left","args":{}},{"action":"up","args":{}}]}

Return no more actions than requested. If you choose `RESET`, it must be the only action and include its required justification in `args`.
