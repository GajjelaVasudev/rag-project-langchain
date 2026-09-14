require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { TextLoader } = require('langchain/document_loaders/fs/text');
const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');
const { GoogleGenerativeAIEmbeddings } = require('@langchain/google-genai');
const { PineconeStore } = require('@langchain/pinecone');
const { Pinecone } = require('@pinecone-database/pinecone');

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const pineconeIndex = pc.index(process.env.PINECONE_INDEX_NAME);

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: 'gemini-embedding-001',
});

async function main() {
  // Sanity check FIRST — confirm the vector size before ingesting everything,
  // same safety habit as before, since this wrapper may not expose the same
  // outputDimensionality option your raw SDK call used.
  const test = await embeddings.embedQuery('test');
  console.log('Embedding dimension:', test.length);
  console.log('Your Pinecone index expects: 1536 (check this matches!)');

  const dataDir = './data';
  const files = fs.readdirSync(dataDir).filter((f) => f.endsWith('.txt'));

  // NOTE: LangChain's splitter counts CHARACTERS, not words (your old chunkText
  // counted words). chunkSize: 2000 chars is roughly equivalent to your old
  // ~400-word chunks; chunkOverlap: 200 chars mirrors your old 50-word overlap.
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 2000,
    chunkOverlap: 200,
  });

  let allDocs = [];

  for (const file of files) {
    const loader = new TextLoader(path.join(dataDir, file));
    const rawDocs = await loader.load(); // [{ pageContent, metadata: { source } }]
    const splitDocs = await splitter.splitDocuments(rawDocs);
    console.log(`${file}: split into ${splitDocs.length} chunks`);
    allDocs.push(...splitDocs);
  }

  // One call replaces: embed each chunk + build records + index.upsert()
  await PineconeStore.fromDocuments(allDocs, embeddings, { pineconeIndex });

  console.log(`Upserted ${allDocs.length} chunks into Pinecone.`);
}

main().catch(console.error);