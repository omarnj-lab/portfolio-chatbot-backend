import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { TaskType } from "@google/generative-ai";
import { readFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);
const model = genAI.getGenerativeModel({ model: "embedding-001" });
const model2 = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(bodyParser.json());
app.use(cors());

// Function Definitions

// Embed retrieval query function
async function embedRetrivalQuery(queryText) {
    const result = await model.embedContent({
        content: { parts: [{ text: queryText }] },
        taskType: TaskType.RETRIEVAL_QUERY,
    });
    const embedding = result.embedding;
    return embedding.values;
}

// Perform query function
async function performQuery(queryText, docs) {
    const queryValues = await embedRetrivalQuery(queryText);

    // Calculate distances
    const distances = docs.map((doc) => ({
        distance: euclideanDistance(doc.values, queryValues),
        text: doc.text,
    }));

    // Sort by distance
    const sortedDocs = distances.sort((a, b) => a.distance - b.distance);

    return sortedDocs.map(doc => doc.text);
}

// Returns Euclidean Distance between 2 vectors
function euclideanDistance(a, b) {
  let sum = 0;
  for (let n = 0; n < a.length; n++) {
    sum += Math.pow(a[n] - b[n], 2);
  }
  return Math.sqrt(sum);
}

// Performs a relevance search for queryText in relation to a known list of embeddings
async function performQuery(queryText, docs) {
  const queryValues = await embedRetrivalQuery(queryText);

  // Calculate distances
  const distances = docs.map((doc) => ({
    distance: euclideanDistance(doc.values, queryValues),
    text: doc.text,
  }));

  // Sort by distance
  const sortedDocs = distances.sort((a, b) => a.distance - b.distance);

  return sortedDocs.map(doc => doc.text);
}

// Generates a final answer using all the relevant documents
async function generateFinalAnswer(queryText, docs) {
  const context = docs.join("\n\n");
  const result = await model2.generateContent(`Question: ${queryText}\n\nContext:\n${context}\n\nAnswer:`)
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

// Define a route for the root URL
app.get('/', (req, res) => {
    res.send('Welcome to the Portfolio Chatbot Backend!');
});

// POST /ask endpoint
app.post('/ask', async (req, res) => {
    const { question } = req.body;
    if (!question) {
        return res.status(400).json({ error: 'Question is required' });
    }

    try {
        const sortedDocs = await performQuery(question, docs);
        const finalAnswer = await generateFinalAnswer(question, sortedDocs);
        res.json({ answer: finalAnswer });
    } catch (error) {
        console.error("Error processing request:", error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
