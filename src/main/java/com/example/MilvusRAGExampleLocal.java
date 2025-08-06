package com.example;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.milvus.MilvusEmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingMatch;

import java.util.List;
import java.util.Scanner;

public class MilvusRAGExampleLocal {
    
    private static EmbeddingStore<TextSegment> embeddingStore;
    private static EmbeddingModel embeddingModel;
    
    public static void main(String[] args) {
        try {
            System.out.println("🚀 Initializing RAG System...");
            
            // 1. Initialize embedding model
            embeddingModel = new AllMiniLmL6V2EmbeddingModel();
            System.out.println("✅ Embedding model loaded");
            
            // 2. Setup Milvus embedding store
            embeddingStore = MilvusEmbeddingStore.builder()
                    .host("localhost")
                    .port(19530)
                    .collectionName("rag_knowledge_base")
                    .dimension(384) // AllMiniLmL6V2 produces 384-dimensional embeddings
                    .build();
            System.out.println("✅ Connected to Milvus");
            
            // 3. Add documents to knowledge base
            addDocumentsToStore();
            
            // 4. Start interactive query loop
            startQueryLoop();
            
        } catch (Exception e) {
            System.err.println("❌ Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void addDocumentsToStore() {
        System.out.println("📚 Adding documents to knowledge base...");
        
        // Sample documents for the knowledge base
        String[] documents = {
            "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms that can learn and improve from experience without being explicitly programmed. It enables computers to learn automatically without human intervention or assistance.",
            
            "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, problem-solving, perception, and language understanding.",
            
            "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to model and understand complex patterns in data. It's particularly effective for tasks like image recognition, natural language processing, and speech recognition.",
            
            "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. It combines computational linguistics with statistical, machine learning, and deep learning models.",
            
            "Computer vision is a field of artificial intelligence that enables computers and systems to derive meaningful information from digital images, videos and other visual inputs. It seeks to automate tasks that the human visual system can do.",
            
            "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information using a connectionist approach to computation.",
            
            "Supervised learning is a machine learning approach where the algorithm learns from labeled training data. The goal is to learn a mapping from inputs to outputs that can generalize to new, unseen data.",
            
            "Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels. It's used for clustering, association rules, and dimensionality reduction.",
            
            "Reinforcement learning is an area of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. It's inspired by behavioral psychology.",
            
            "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data. It combines domain expertise, programming skills, and knowledge of mathematics and statistics.",
            
            "Big data refers to extremely large datasets that may be analyzed computationally to reveal patterns, trends, and associations, especially relating to human behavior and interactions. It's characterized by volume, velocity, and variety.",
            
            "Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user. It relies on sharing of resources to achieve coherence and economies of scale."
        };
        
        // Document splitter for chunking
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 50);
        
        int totalSegments = 0;
        for (String docText : documents) {
            // Create document
            Document document = Document.from(docText);
            
            // Split document into segments
            List<TextSegment> segments = splitter.split(document);
            
            // Generate embeddings and store
            for (TextSegment segment : segments) {
                Embedding embedding = embeddingModel.embed(segment).content();
                embeddingStore.add(embedding, segment);
                totalSegments++;
            }
        }
        
        System.out.println("✅ Added " + documents.length + " documents (" + totalSegments + " segments) to knowledge base");
    }
    
    private static void startQueryLoop() {
        System.out.println("\n🎯 RAG System Ready!");
        System.out.println("Ask questions about AI, ML, Data Science, or related topics.");
        System.out.println("Type 'quit' or 'exit' to stop.\n");
        
        Scanner scanner = new Scanner(System.in);
        
        while (true) {
            System.out.print("❓ Your question: ");
            String query = scanner.nextLine().trim();
            
            if (query.equalsIgnoreCase("quit") || query.equalsIgnoreCase("exit")) {
                System.out.println("👋 Goodbye!");
                break;
            }
            
            if (query.isEmpty()) {
                continue;
            }
            
            try {
                // Retrieve relevant documents
                List<EmbeddingMatch<TextSegment>> relevantDocuments = retrieveRelevantDocuments(query, 3);
                
                if (relevantDocuments.isEmpty()) {
                    System.out.println("🤷 No relevant documents found for your query.\n");
                    continue;
                }
                
                // Display results
                System.out.println("\n📄 Retrieved Information:");
                System.out.println("=" + "=".repeat(50));
                
                for (int i = 0; i < relevantDocuments.size(); i++) {
                    EmbeddingMatch<TextSegment> match = relevantDocuments.get(i);
                    System.out.printf("📌 Result %d (Similarity: %.3f):\n", i + 1, match.score());
                    System.out.println(match.embedded().text());
                    System.out.println();
                }
                
                // Simple answer generation (without LLM)
                System.out.println("🤖 Generated Answer:");
                System.out.println("-".repeat(30));
                generateSimpleAnswer(query, relevantDocuments);
                System.out.println();
                
            } catch (Exception e) {
                System.err.println("❌ Error processing query: " + e.getMessage());
            }
        }
        
        scanner.close();
    }
    
    private static List<EmbeddingMatch<TextSegment>> retrieveRelevantDocuments(String query, int maxResults) {
        // Embed the query
        Embedding queryEmbedding = embeddingModel.embed(query).content();
        
        // Search for similar documents
        return embeddingStore.findRelevant(queryEmbedding, maxResults, 0.5);
    }
    
    private static void generateSimpleAnswer(String query, List<EmbeddingMatch<TextSegment>> relevantDocs) {
        // Simple rule-based answer generation
        String queryLower = query.toLowerCase();
        
        if (queryLower.contains("what is") || queryLower.contains("define")) {
            System.out.println("Based on the retrieved documents:");
            if (!relevantDocs.isEmpty()) {
                // Use the most relevant document for definition
                String mostRelevant = relevantDocs.get(0).embedded().text();
                System.out.println(mostRelevant);
            }
        } else if (queryLower.contains("how") || queryLower.contains("explain")) {
            System.out.println("Here's an explanation based on the knowledge base:");
            for (EmbeddingMatch<TextSegment> doc : relevantDocs) {
                if (doc.score() > 0.7) { // High similarity threshold
                    System.out.println("• " + doc.embedded().text());
                    break;
                }
            }
        } else if (queryLower.contains("difference") || queryLower.contains("compare")) {
            System.out.println("Here are the relevant comparisons I found:");
            for (int i = 0; i < Math.min(2, relevantDocs.size()); i++) {
                System.out.println("• " + relevantDocs.get(i).embedded().text());
            }
        } else {
            System.out.println("Here are the most relevant pieces of information I found:");
            for (int i = 0; i < Math.min(2, relevantDocs.size()); i++) {
                System.out.println("• " + relevantDocs.get(i).embedded().text());
            }
        }
        
        System.out.println("\n💡 For more detailed answers, consider integrating with a language model like OpenAI GPT!");
    }
}