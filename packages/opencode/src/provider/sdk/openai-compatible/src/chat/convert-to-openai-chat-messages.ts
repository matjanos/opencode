import {
  type LanguageModelV2CallWarning,
  type LanguageModelV2Prompt,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider"
import { convertToBase64 } from "@ai-sdk/provider-utils"

export async function convertToOpenAIChatMessages({ prompt }: { prompt: LanguageModelV2Prompt }): Promise<{
  messages: Array<any>
  warnings: Array<LanguageModelV2CallWarning>
}> {
  const messages: Array<any> = []
  const warnings: Array<LanguageModelV2CallWarning> = []

  for (const { role, content } of prompt) {
    switch (role) {
      case "system": {
        messages.push({ role: "system", content })
        break
      }

      case "user": {
        messages.push({
          role: "user",
          content: content.map((part) => {
            switch (part.type) {
              case "text": {
                return { type: "text", text: part.text }
              }
              case "file": {
                if (part.mediaType.startsWith("image/")) {
                  const mediaType = part.mediaType === "image/*" ? "image/jpeg" : part.mediaType
                  return {
                    type: "image_url",
                    image_url: {
                      url:
                        part.data instanceof URL
                          ? part.data.toString()
                          : `data:${mediaType};base64,${convertToBase64(part.data)}`,
                    },
                  }
                }
                throw new UnsupportedFunctionalityError({
                  functionality: `file part media type ${part.mediaType}`,
                })
              }
            }
          }),
        })
        break
      }

      case "assistant": {
        let textContent = ""
        let reasoningContent = ""
        const toolCalls: any[] = []

        for (const part of content) {
          switch (part.type) {
            case "text": {
              textContent += part.text
              break
            }
            case "reasoning": {
              reasoningContent += part.text
              break
            }
            case "tool-call": {
              toolCalls.push({
                id: part.toolCallId,
                type: "function",
                function: {
                  name: part.toolName,
                  arguments: JSON.stringify(part.input),
                },
              })
              break
            }
          }
        }

        const message: any = {
          role: "assistant",
          content: textContent || null, // OpenAI allows null content if tool_calls present
        }

        if (reasoningContent) {
          message.reasoning_content = reasoningContent
        }

        if (toolCalls.length > 0) {
          message.tool_calls = toolCalls
        }

        messages.push(message)
        break
      }

      case "tool": {
        for (const part of content) {
          messages.push({
            role: "tool",
            tool_call_id: part.toolCallId,
            content: JSON.stringify(part.output),
          })
        }
        break
      }

      default: {
        const _exhaustiveCheck: never = role
        throw new Error(`Unsupported role: ${_exhaustiveCheck}`)
      }
    }
  }

  return { messages, warnings }
}
