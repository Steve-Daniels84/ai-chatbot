import {
  customProvider,
  extractReasoningMiddleware,
  wrapLanguageModel,
} from 'ai';
import { xai } from '@ai-sdk/xai';
import {
  artifactModel,
  chatModel,
  reasoningModel,
  titleModel,
} from './models.test';
import { isTestEnvironment } from '../constants';
import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from '@aws-sdk/client-bedrock-runtime';
import type {
  LanguageModelV2Content,
  LanguageModelV2FinishReason,
} from '@ai-sdk/provider';
import type { LanguageModelV2StreamPart } from '@ai-sdk/provider';

type LanguageModelV2Text = {
  type: 'text';
  text: string;
  providerMetadata?: any;
};

function extractPromptString(prompt: any): string {
  if (typeof prompt === 'string') return prompt;
  if (Array.isArray(prompt)) return prompt.map(extractPromptString).join('\n');
  if (typeof prompt === 'object' && prompt !== null) {
    // If it's a message object with 'content' or 'text', extract those
    if ('content' in prompt && typeof prompt.content === 'string') {
      return prompt.content;
    }
    if ('text' in prompt && typeof prompt.text === 'string') {
      return prompt.text;
    }
    // If 'content' is an array, join all text recursively
    if ('content' in prompt && Array.isArray(prompt.content)) {
      return prompt.content.map(extractPromptString).join('\n');
    }
    // Otherwise, join all string values in the object
    return Object.values(prompt)
      .map((v) => (typeof v === 'string' ? v : extractPromptString(v)))
      .join('\n');
  }
  return '';
}

const bedrockClient = new BedrockRuntimeClient({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
  },
});

async function callClaude37Bedrock(prompt: string) {
  const modelId =
    process.env.BEDROCK_CLAUDE_MODEL_ID ||
    'anthropic.claude-3-sonnet-20240229-v1:0';
  const body = {
    anthropic_version: 'bedrock-2023-05-31',
    max_tokens: 1024,
    messages: [{ role: 'user', content: prompt }],
  };
  try {
    console.log('[Bedrock Claude 3.7] Prompt:', prompt);
    const command = new InvokeModelCommand({
      modelId,
      contentType: 'application/json',
      accept: 'application/json',
      body: JSON.stringify(body),
    });
    const response = await bedrockClient.send(command);
    const responseBody = await response.body.transformToString();
    const parsed = JSON.parse(responseBody);
    console.log('[Bedrock Claude 3.7] Response:', parsed);
    // Claude 3.7 returns content in a 'content' array
    return parsed.content?.map((c: any) => c.text).join(' ') || '';
  } catch (err) {
    console.error('[Bedrock Claude 3.7] Error:', err);
    return '[Error: Failed to get response from Bedrock Claude 3.7]';
  }
}

const bedrockClaude37Provider = {
  specificationVersion: 'v2' as const,
  provider: 'bedrock',
  modelId:
    process.env.BEDROCK_CLAUDE_MODEL_ID ||
    'anthropic.claude-3-sonnet-20240229-v1:0',
  supportedUrls: {},
  doGenerate: async (options: any) => {
    const prompt = extractPromptString(options.prompt);
    const text = String(await callClaude37Bedrock(prompt));
    return {
      content: [{ type: 'text', text } as LanguageModelV2Content],
      finishReason: 'stop' as LanguageModelV2FinishReason,
      usage: {
        inputTokens: undefined,
        outputTokens: undefined,
        totalTokens: undefined,
      },
      rawCall: {},
      warnings: [],
    };
  },
  doStream: async (options: any) => {
    const prompt = extractPromptString(options.prompt);
    const text = String(await callClaude37Bedrock(prompt));
    // Create a ReadableStream that emits the text as a single delta
    const stream = new ReadableStream<LanguageModelV2StreamPart>({
      start(controller) {
        controller.enqueue({ type: 'text-start', id: '1' });
        controller.enqueue({ type: 'text-delta', id: '1', delta: text });
        controller.enqueue({ type: 'text-end', id: '1' });
        controller.enqueue({
          type: 'finish',
          usage: {
            inputTokens: undefined,
            outputTokens: undefined,
            totalTokens: undefined,
          },
          finishReason: 'stop' as LanguageModelV2FinishReason,
        });
        controller.close();
      },
    });
    return { stream };
  },
};

export const myProvider = isTestEnvironment
  ? customProvider({
      languageModels: {
        'chat-model': chatModel,
        'chat-model-reasoning': reasoningModel,
        'title-model': titleModel,
        'artifact-model': artifactModel,
        // Add for test env if needed
        'claude-3-7-bedrock': bedrockClaude37Provider,
      },
    })
  : customProvider({
      languageModels: {
        'chat-model': xai('grok-2-vision-1212'),
        'chat-model-reasoning': wrapLanguageModel({
          model: xai('grok-3-mini-beta'),
          middleware: extractReasoningMiddleware({ tagName: 'think' }),
        }),
        'title-model': xai('grok-2-1212'),
        'artifact-model': xai('grok-2-1212'),
        'claude-3-7-bedrock': bedrockClaude37Provider,
      },
      imageModels: {
        'small-model': xai.imageModel('grok-2-image'),
      },
    });
