# Exploring GPT5 and Responses API

I'd like to explore OpenAI's python Responses API, including using some of the new features released with GPT5. Given most of OpenAI's current flagship models are being deprecated, I will focus exclusively on features compatable with GPT5 for this exercise.

Here are some scenarios I'd like to document for later reference:
- [x] Generating responses (hello gpt5)
- [x] Providing developer prompts
- [x] Getting back reasoning summary
- [x] Changing reasoning level
- [x] Tool calls
- [ ] Structured outputs
- [ ] Getting token counts
- [ ] Change verbosity
- [ ] Provide image context
- [ ] Multi-turn conversation


## Hello GPT5

The most minimal of model prompting in [hellogpt5.py](scripts/hellogpt5.py).

```python
client = OpenAI()

# Get a simple hello gpt-5 response
response = client.responses.create(
    model="gpt-5-nano", input=[{"role": "user", "content": "Hello, gpt-5!"}]
)
print(response.output_text)
```

Output:
```
Hello! I’m ChatGPT. I’m not GPT-5, but I’m here to help with questions, writing, coding, planning, learning, and more. What would you like to do today?

If you’re unsure, tell me your goal and I’ll suggest a path. I can:
- Explain concepts simply or in depth
- Draft or edit text (emails, essays, resumes)
- Help debug code or explain algorithms
- Brainstorm ideas or plan a project
- Solve math problems step by step
- Analyze an image or describe it (if you upload one)

What can I assist you with?
```

Interestingly, the model doesn't seem to know it is GPT5.


## Exploring the instruction hierarchy

In 2024, OpenAI added a "developer" role to its instruction hierarchy, with their models trained to maximally comply with instructions provided by the "system", followed by the "developer", then finally the "user". The new "developer" role effectively superceded the previous intent of the "system" role. This enabled OpenAI to maintain control of a base layer "system" instruction which is inaccessible to the application developer.

However, this is a bit confusing because the Responses API still supports providing a "system" instruction. This gets even more confusing when you consider they have also added an "instruction" paramater to the responses API which also serves an equivalent purpose to the "developer" role.

To figure out what is going on, let's run some [tests where we provide conflicting instructions](scripts/instructionhierarchy.py) using the system and developer roles, along with the instruction parameter.

```python
print(
    "Testing prompt hierachy when given a conflicting 'system' followed by 'developer' prompt."
)
for i in range(5):
    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": "Reply with the message `green`.",
            },
            {
                "role": "developer",
                "content": "Reply with the message `red`.",
            },
            {
                "role": "user",
                "content": "Reply with the message `yellow`.",
            },
        ],
    )
    print(f"Trial {i}: {response.output_text}")


print(
    "\nTesting prompt hierachy when given a conflicting 'developer' followed by 'system' prompt."
)
for i in range(5):
    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "developer",
                "content": "Reply with the message `red`.",
            },
            {
                "role": "system",
                "content": "Reply with the message `green`.",
            },
            {
                "role": "user",
                "content": "Reply with the message `yellow`.",
            },
        ],
    )
    print(f"Trial {i}: {response.output_text}")


print(
    "\nTesting prompt hierachy when given a conflicting 'instruction', 'developer' and 'system' prompt."
)
for i in range(5):
    response = client.responses.create(
        model="gpt-5",
        instructions="Reply with the message `blue`.",
        input=[
            {
                "role": "system",
                "content": "Reply with the message `green`.",
            },
            {
                "role": "developer",
                "content": "Reply with the message `red`.",
            },
            {
                "role": "user",
                "content": "Reply with the message `yellow`.",
            },
        ],
    )
    print(f"Trial {i}: {response.output_text}")


print("\nConfirm `instructions` prompt actually works.")
for i in range(5):
    response = client.responses.create(
        model="gpt-5",
        instructions="Reply with the message `blue`.",
        input=[
            {
                "role": "user",
                "content": "Reply with the message `yellow`.",
            },
        ],
    )
    print(f"Trial {i}: {response.output_text}")
```

Output:
```
Testing prompt hierachy when given a conflicting 'system' followed by 'developer' prompt.
Trial 0: red
Trial 1: red
Trial 2: red
Trial 3: red
Trial 4: red

Testing prompt hierachy when given a conflicting 'developer' followed by 'system' prompt.
Trial 0: green
Trial 1: green
Trial 2: green
Trial 3: green
Trial 4: green

Testing prompt hierachy when given a conflicting 'instruction', 'developer' and 'system' prompt.
Trial 0: red
Trial 1: red
Trial 2: red
Trial 3: red
Trial 4: red

Confirm `instructions` prompt actually works.
Trial 0: blue
Trial 1: blue
Trial 2: blue
Trial 3: blue
Trial 4: blue
```

In the case of conflicts, we see that the model simply follows the latest instruction (amongst the system/developer/instruction prompts). It appears that each of these methods are roughly equivalent. I suspect that the "system" role is maintained in the API for backwards compatability but under the hood, it's treated equivalently to "developer". I assume "instructions" are also equivalent.


## Reasoning level and reasoning summaries

To set reasoning level and include reasoning summaries (see [reasoningsummary.py](scripts/reasoningsummary.py)):

```python
response = client.responses.create(
    model="gpt-5-nano",
    reasoning={"effort": "low", "summary": "detailed"},
    input=[{"role": "user", "content": "Hello, gpt-5!"}],
)
for el in response.output:
    if el.type == "reasoning":
        print("Summary:")
        print(el)
        break
    raise Exception("Did not find a reasoning item")
print(f"Response: {response.output_text}")
```

## Tool calling

### Specifying tool usage behaviour

Let's examine setting up different model behaviours for tool calling:
- call exactly one tool
- call multiple tools in parallel
- call any number of tools (zero, one, many)

Per [scripts/toolcalling.py], we can specify the behaviour we want by manipulating the
`parallel_tool_calls` and `tool_choice` parameters:

```python
response = client.responses.create(
    model="gpt-5-nano",
    input=[{"role": "user", "content": "Hello."}],
    tools=tools,
    parallel_tool_calls=False,  # Forces no more than one tool call | Or can set to True to allow multiple tool calls
    tool_choice="required",  # Forces a tool call | Or can set to `auto` to make tool calling optional
)
```

### Returning tool call results

In [scripts/toolcallingloop.py], we demonstrate how to call tools requested by the model and return the results.