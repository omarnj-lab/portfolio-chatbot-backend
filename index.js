import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';
import faiss from 'faiss'; // Make sure faiss is installed or use appropriate import
import { GoogleGenerativeAI } from "@google/generative-ai";
import { TaskType } from "@google/generative-ai";
import { readFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Initialize Google Generative AI
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);

const model = genAI.getGenerativeModel({ model: "text-embedding-004" });
const model2 = genAI.getGenerativeModel({
  model: "gemini-1.5-flash",
  systemInstruction: "You are Omar's CV Chatbot assistant that retrieves information about my CV. Your Answer should be precise and attractive. Always answer from the context and do not improvise. Provide answers in bullets and in an organized way",
});

// Get the directory name of the current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Initialize Express server
const app = express();
app.use(bodyParser.json());
app.use(cors());

// Utility function for cosine similarity
function cosineSimilarity(a, b) {
  let dotProduct = 0.0;
  let aMagnitude = 0.0;
  let bMagnitude = 0.0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    aMagnitude += a[i] * a[i];
    bMagnitude += b[i] * b[i];
  }
  aMagnitude = Math.sqrt(aMagnitude);
  bMagnitude = Math.sqrt(bMagnitude);
  return dotProduct / (aMagnitude * bMagnitude);
}

// Function to embed retrieval query
async function embedRetrivalQuery(queryText) {
  const result = await model.embedContent({
    content: { parts: [{ text: queryText }] },
    taskType: TaskType.RETRIEVAL_QUERY,
  });
  return result.embedding.values;
}

// Function to embed retrieval documents with batching
async function embedRetrivalDocuments(docTexts) {
  const batchSize = 100; // API limit is 100 requests per batch
  const embeddings = [];

  for (let i = 0; i < docTexts.length; i += batchSize) {
    const batch = docTexts.slice(i, i + batchSize);

    const result = await model.batchEmbedContents({
      requests: batch.map((t) => ({
        content: { parts: [{ text: t }] },
        taskType: TaskType.RETRIEVAL_DOCUMENT,
      })),
    });

    embeddings.push(...result.embeddings.map((e, index) => ({ text: batch[index], values: e.values })));
  }

  return embeddings;
}

// Initialize FAISS Index
async function initializeFaissIndex(embeddings) {
  const dimension = embeddings[0].values.length; // Assuming all embeddings have the same dimension
  const index = new faiss.IndexFlatL2(dimension); // Using L2 distance

  // Add all embeddings to the index
  embeddings.forEach((embedding) => {
    index.add(faiss.floatArray(embedding.values));
  });

  return index;
}

// Function to perform a relevance search for queryText in relation to a known list of embeddings
async function performQuery(queryText, docs, index) {
  const queryValues = await embedRetrivalQuery(queryText);

  // Perform search using FAISS
  const k = 10; // Number of nearest neighbors to retrieve
  const distances = new Float32Array(k);
  const labels = new Int32Array(k);
  index.search(faiss.floatArray(queryValues), k, distances, labels);

  // Retrieve the corresponding documents
  const sortedDocs = [];
  for (let i = 0; i < k; i++) {
    sortedDocs.push(docs[labels[i]].text);
  }

  return sortedDocs;
}

// Function to generate a final answer using all the relevant documents
async function generateFinalAnswer(queryText, docs) {
  const context = docs.join("\n\n");
  const result = await model2.generateContent(`Question: ${queryText}\n\nContext:\n${context}\n\nAnswer:`);
  const response = await result.response;
  const text = await response.text();

  // Clean up the final answer
  const cleanedText = text.replace(/\*\*/g, '').replace(/\n/g, ' ');
  return cleanedText;
}

// Load the document texts from embeddings.txt
const txtPath = path.resolve(__dirname, 'embeddings.txt');
const loadEmbeddingsTxt = () => {
  const fileContent = readFileSync(txtPath, 'utf-8');
  const docs = fileContent.split('\n').filter(line => line.trim() !== '');
  return docs;
};
const docTexts = loadEmbeddingsTxt();

// Precompute embeddings for our documents and store in FAISS
let docs = [];
let faissIndex;
embedRetrivalDocuments(docTexts).then(async (precomputedDocs) => {
  docs = precomputedDocs;
  faissIndex = await initializeFaissIndex(precomputedDocs);
});

// Define the POST endpoint
app.post('/ask', async (req, res) => {
  const { question } = req.body;
  console.log("Received question:", question);
  if (!question) {
    return res.status(400).json({ error: 'Question is required' });
  }

  try {
    // Use retrieval query embeddings to find most relevant documents via FAISS
    const sortedDocs = await performQuery(question, docs, faissIndex);

    // Generate a final answer using all the relevant documents
    const finalAnswer = await generateFinalAnswer(question, sortedDocs);
    res.json({ answer: finalAnswer });
  } catch (error) {
    console.error("Error processing request:", error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Start the server
const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
