import { type LanguageModelV2, type LanguageModelV2StreamPart } from "@ai-sdk/provider"
import {
  createJsonResponseHandler,
  createEventSourceResponseHandler,
  postJsonToApi,
  combineHeaders,
} from "@ai-sdk/provider-utils"
import { z } from "zod"
import { convertToOpenAIChatMessages } from "./convert-to-openai-chat-messages"
import type { OpenAICompatibleProviderSettings } from "../openai-compatible-provider"

const chatChunkSchema = z.object({
  id: z.string(),
  choices: z.array(
    z.object({
      delta: z.object({
        role: z.string().optional(),
        content: z.string().nullable().optional(),
        reasoning_content: z.string().optional(),
        tool_calls: z
          .array(
            z.object({
              index: z.number(),
              id: z.string().optional(),
              type: z.literal("function").optional(),
              function: z
                .object({
                  name: z.string().optional(),
                  arguments: z.string().optional(),
                })
                .optional(),
            }),
          )
          .optional(),
      }),
      finish_reason: z.string().nullable().optional(),
    }),
  ),
  usage: z
    .object({
      prompt_tokens: z.number(),
      completion_tokens: z.number(),
    })
    .optional(),
})

export class OpenAICompatibleChatLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = "v1"
  readonly modelId: string
  readonly config: OpenAICompatibleProviderSettings & {
    url: (options: { path: string; modelId: string }) => string
    headers: () => Record<string, string | undefined>
  }

  constructor(
    modelId: string,
    config: OpenAICompatibleProviderSettings & {
      url: (options: { path: string; modelId: string }) => string
      headers: () => Record<string, string | undefined>
    },
  ) {
    this.modelId = modelId
    this.config = config
  }

  get provider(): string {
    return this.config.name ?? "openai-compatible"
  }

  async doGenerate(options: Parameters<LanguageModelV2["doGenerate"]>[0]) {
    const { messages, warnings } = await convertToOpenAIChatMessages({
      prompt: options.prompt,
    })

    const body = {
      model: this.modelId,
      messages,
      temperature: options.temperature,
      top_p: options.topP,
      max_tokens: options.maxTokens,
      presence_penalty: options.presencePenalty,
      frequency_penalty: options.frequencyPenalty,
      seed: options.seed,
    }

    const url = this.config.url({
      path: "/chat/completions",
      modelId: this.modelId,
    })

    const { value: response } = await postJsonToApi({
      url,
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: createJsonResponseHandler(z.any()),
      successfulResponseHandler: createJsonResponseHandler(
        z.object({
          id: z.string(),
          choices: z.array(
            z.object({
              message: z.object({
                role: z.string(),
                content: z.string().nullable(),
                tool_calls: z
                  .array(
                    z.object({
                      id: z.string(),
                      type: z.literal("function"),
                      function: z.object({
                        name: z.string(),
                        arguments: z.string(),
                      }),
                    }),
                  )
                  .optional(),
                reasoning_content: z.string().optional(),
              }),
              finish_reason: z.string().nullable(),
            }),
          ),
          usage: z
            .object({
              prompt_tokens: z.number(),
              completion_tokens: z.number(),
            })
            .optional(),
        }),
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    })

    const choice = response.choices[0]
    const content: any[] = []

    if (choice.message.reasoning_content) {
      content.push({ type: "reasoning", text: choice.message.reasoning_content })
    }

    if (choice.message.content) {
      content.push({ type: "text", text: choice.message.content })
    }

    if (choice.message.tool_calls) {
      for (const toolCall of choice.message.tool_calls) {
        content.push({
          type: "tool-call",
          toolCallId: toolCall.id,
          toolName: toolCall.function.name,
          args: toolCall.function.arguments,
        })
      }
    }

    return {
      text: choice.message.content ?? undefined,
      finishReason: choice.finish_reason as any,
      usage: {
        promptTokens: response.usage?.prompt_tokens ?? NaN,
        completionTokens: response.usage?.completion_tokens ?? NaN,
      },
      rawCall: { rawPrompt: messages, rawSettings: body },
      rawResponse: response,
      warnings,
    }
  }

  async doStream(options: Parameters<LanguageModelV2["doStream"]>[0]) {
    const { messages, warnings } = await convertToOpenAIChatMessages({
      prompt: options.prompt,
    })

    const body = {
      model: this.modelId,
      messages,
      stream: true,
      temperature: options.temperature,
      top_p: options.topP,
      max_tokens: options.maxTokens,
      presence_penalty: options.presencePenalty,
      frequency_penalty: options.frequencyPenalty,
      seed: options.seed,
    }

    const url = this.config.url({
      path: "/chat/completions",
      modelId: this.modelId,
    })

    const { value: response } = await postJsonToApi({
      url,
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: createJsonResponseHandler(z.any()),
      successfulResponseHandler: createEventSourceResponseHandler(chatChunkSchema),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    })

    let finishReason: any = "unknown"
    let usage: { promptTokens: number; completionTokens: number } | undefined = undefined
    let reasoningPartId: string | undefined

    return {
      stream: response.pipeThrough(
        new TransformStream<
          { success: true; value: z.infer<typeof chatChunkSchema> } | { success: false; error: any },
          LanguageModelV2StreamPart
        >({
          start(controller) {
            controller.enqueue({ type: "stream-start", warnings })
          },
          transform(chunk, controller) {
            if (!chunk.success) {
              controller.enqueue({ type: "error", error: chunk.error })
              return
            }
            const value = chunk.value
            if (value.usage) {
              usage = {
                promptTokens: value.usage.prompt_tokens,
                completionTokens: value.usage.completion_tokens,
              }
            }
            if (value.choices.length > 0) {
              const choice = value.choices[0]
              if (choice.finish_reason) {
                finishReason = choice.finish_reason
              }
              const delta = choice.delta
              if (delta.reasoning_content) {
                if (!reasoningPartId) {
                  reasoningPartId = "reasoning"
                  controller.enqueue({
                    type: "reasoning-start",
                    id: reasoningPartId,
                  } as any)
                }

                controller.enqueue({
                  type: "reasoning-delta",
                  id: reasoningPartId,
                  textDelta: delta.reasoning_content,
                } as any)
              }
              if (delta.content) {
                controller.enqueue({
                  type: "text-delta",
                  textDelta: delta.content,
                })
              }
              if (delta.tool_calls) {
                for (const toolCall of delta.tool_calls) {
                  controller.enqueue({
                    type: "tool-call-delta",
                    toolCallId: toolCall.id ?? "",
                    toolName: toolCall.function?.name ?? "",
                    argsTextDelta: toolCall.function?.arguments ?? "",
                  })
                }
              }
            }
          },
          flush(controller) {
            controller.enqueue({
              type: "finish",
              finishReason,
              usage: usage ?? { promptTokens: NaN, completionTokens: NaN },
            })
          },
        }),
      ),
      rawCall: { rawPrompt: messages, rawSettings: body },
      rawResponse: response,
      warnings,
    }
  }
}
