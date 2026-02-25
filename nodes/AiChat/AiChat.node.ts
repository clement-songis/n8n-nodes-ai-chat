import type {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';
import { NodeConnectionTypes, NodeOperationError } from 'n8n-workflow';
/* eslint-disable @n8n/community-nodes/no-restricted-imports */
import { isChatInstance } from '@n8n/ai-utilities';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { createStuffDocumentsChain } from '@langchain/classic/chains/combine_documents';
import { createRetrievalChain } from '@langchain/classic/chains/retrieval';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseLanguageModel } from '@langchain/core/language_models/base';
import type { BaseChatMemory } from '@langchain/classic/memory';
import type { VectorStore } from '@langchain/core/vectorstores';
/* eslint-enable @n8n/community-nodes/no-restricted-imports */

const SYSTEM_MESSAGE = 'You are a helpful assistant.';

export class AiChat implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Ai Chat',
		name: 'aiChat',
		icon: { light: 'file:ai-chat.svg', dark: 'file:ai-chat.dark.svg' },
		group: ['transform'],
		version: [1],
		description: 'An Ai chat with memory and vector store',
		defaults: {
			name: 'Ai Chat',
		},
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Agents', 'Root Nodes'],
			},
		},
		inputs: [
			NodeConnectionTypes.Main,
			{
				displayName: 'Model',
				maxConnections: 1,
				type: NodeConnectionTypes.AiLanguageModel,
				required: true,
			},
			{
				displayName: 'Vector Store',
				maxConnections: 1,
				type: NodeConnectionTypes.AiVectorStore,
				required: true,
			},
			{
				displayName: 'Memory',
				maxConnections: 1,
				type: NodeConnectionTypes.AiMemory,
				required: true,
			},
		],
		outputs: [NodeConnectionTypes.Main],
		builderHint: {
			inputs: {
				ai_languageModel: { required: true },
				ai_memory: { required: true },
				ai_vectorStore: { required: true },
			},
		},
		usableAsTool: true,
		properties: [
			{
				displayName: 'Prompt (User Message)',
				name: 'chatInput',
				type: 'string',
				required: true,
				default: '={{ $json.chatInput }}',
				placeholder: 'e.g. Hello, how can you help me?',
				typeOptions: {
					rows: 2,
				},
				builderHint: {
					message:
						'Use expressions to include dynamic data from previous nodes (e.g., "={{ $json.input }}"). Static text prompts ignore incoming data.',
				},
			},
			{
				displayName: 'Max Chat History Messages',
				name: 'maxChatHistoryMessages',
				type: 'number',
				default: 4,
				description: 'Maximum chat history messages included in prompt',
			},
			{
				displayName: 'Max Vector Store Results',
				name: 'maxVectorStoreResults',
				type: 'number',
				default: 4,
				description: 'Maximum vector store results included in prompt',
			},
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				default: {},
				placeholder: 'Add Option',
				options: [
					{
						displayName: 'System Message',
						name: 'systemMessage',
						type: 'string',
						default: SYSTEM_MESSAGE,
						description:
							'The message that will be sent to the agent before the conversation starts',
						builderHint: {
							message:
								"Must include: agent's purpose, exact names of connected tools, and response instructions",
						},
						typeOptions: {
							rows: 6,
						},
					},
				],
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();

		let item: INodeExecutionData;
		let input: string;
		let systemMessage: string;
		let maxVectorStoreResults: number;

		for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
			try {
				input = this.getNodeParameter('chatInput', itemIndex, '') as string;
				systemMessage = this.getNodeParameter(
					'options.systemMessage',
					itemIndex,
					SYSTEM_MESSAGE,
				) as string;
				maxVectorStoreResults = this.getNodeParameter(
					'maxVectorStoreResults',
					itemIndex,
					4,
				) as number;

				item = items[itemIndex];

				const llm = (await this.getInputConnectionData(
					NodeConnectionTypes.AiLanguageModel,
					itemIndex,
				)) as BaseLanguageModel;

				if (!isChatInstance(llm)) {
					throw new NodeOperationError(this.getNode(), 'Requires Chat Model');
				}

				const memory = (await this.getInputConnectionData(
					NodeConnectionTypes.AiMemory,
					itemIndex,
				)) as BaseChatMemory;

				const vectorStore = (await this.getInputConnectionData(
					NodeConnectionTypes.AiVectorStore,
					itemIndex,
				)) as VectorStore;

				const retriever = vectorStore.asRetriever(maxVectorStoreResults);

				const memoryVariables = await memory.loadMemoryVariables({});

				const prompt = ChatPromptTemplate.fromMessages([
					['system', '{system_message}\n----------------\nContext: {context}'],
					['placeholder', '{chat_history}'],
					['human', '{input}'],
				]);

				const combineDocsChain = await createStuffDocumentsChain({
					llm,
					prompt,
				});

				const retrievalChain = await createRetrievalChain({
					combineDocsChain,
					retriever,
				});

				const eventStream = retrievalChain.streamEvents(
					{
						input,
						system_message: systemMessage || SYSTEM_MESSAGE,
						chat_history: memoryVariables['chat_history'] || [],
					},
					{
						version: 'v2',
						signal: this.getExecutionCancelSignal(),
					},
				);

				let output = '';

				this.sendChunk('begin', 0);
				for await (const event of eventStream) {
					if (event.event === 'on_chat_model_stream') {
						const chunk = event.data?.chunk;
						if (chunk?.content) {
							const chunkContent = chunk.content;
							let chunkText = '';
							if (Array.isArray(chunkContent)) {
								for (const message of chunkContent) {
									if (message?.type === 'text') {
										chunkText += message?.text;
									}
								}
							} else if (typeof chunkContent === 'string') {
								chunkText = chunkContent;
							}
							this.sendChunk('item', 0, chunkText);

							output += chunkText;
						}
					}
				}
				this.sendChunk('end', 0);

				await memory.chatHistory.addMessages([new HumanMessage(input), new AIMessage(output)]);

				item.json = { output };
			} catch (error) {
				if (this.continueOnFail()) {
					items.push({ json: this.getInputData(itemIndex)[0].json, error, pairedItem: itemIndex });
				} else {
					if (error.context) {
						error.context.itemIndex = itemIndex;
						throw error;
					}
					throw new NodeOperationError(this.getNode(), error, {
						itemIndex,
					});
				}
			}
		}

		return [items];
	}
}
