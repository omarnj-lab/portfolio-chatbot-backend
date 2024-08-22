import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors'; // Import cors middleware
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
  systemInstruction: "You are Omar's CV Chatbot assistant that retrieves information about my CV. Your Answer should be precise and attractive. Always answer from the context and do not improvise. Provide answers in bullets and in an organized way.",
});

// Get the directory name of the current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Initialize Express server
const app = express();
app.use(bodyParser.json());
app.use(cors());

// Function to embed retrieval query
async function embedRetrivalQuery(queryText) {
  const result = await model.embedContent({
    content: { parts: [{ text: queryText }] },
    taskType: TaskType.RETRIEVAL_QUERY,
  });
  const embedding = result.embedding;
  return embedding.values;
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

// Function to calculate Cosine Similarity between 2 vectors
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

// Function to perform a relevance search for queryText in relation to a known list of embeddings
async function performQuery(queryText, docs) {
  const queryValues = await embedRetrivalQuery(queryText);

  // Calculate distances
  const distances = docs.map((doc) => ({
    distance: cosineSimilarity(doc.values, queryValues),
    text: doc.text,
  }));

  // Sort by distance
  const sortedDocs = distances.sort((a, b) => b.distance - a.distance);

  return sortedDocs.map(doc => doc.text);
}

// Function to generate a final answer using all the relevant documents
async function generateFinalAnswer(queryText, docs) {
  const context = docs.join("\n\n");
  const result = await model2.generateContent(`Question: ${queryText}\n\nContext:\n${context}\n\nAnswer:`);
  const response = await result.response;
  const text = await response.text();

  // Remove ** and \n from the final answer
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

// Precompute embeddings for our documents
let docs = [];
embedRetrivalDocuments(docTexts).then((precomputedDocs) => {
  docs = precomputedDocs;
});

// Define the POST endpoint
app.post('/ask', async (req, res) => {
  const { question } = req.body;
  console.log("Received question:", question);
  if (!question) {
    return res.status(400).json({ error: 'Question is required' });
  }

  try {
    // Use retrieval query embeddings to find most relevant documents
    const sortedDocs = await performQuery(question, docs);

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
