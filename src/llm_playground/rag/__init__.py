"""RAG: chunking (prose vs AST-aware), embeddings, vector stores, retrieval.

Topic 1 of the interview sprint. Key ideas demonstrated here:
    - Chunking strategy depends on content type: prose uses sentence-aware
      splitting; code uses AST-aware splitting so functions/classes stay intact.
    - Vector-store choice is a real architecture decision: Chroma (in-process,
      fast to start) vs pgvector (vectors next to relational data, SQL-native).
    - Everything is behind small interfaces (Protocols) so the pieces compose
      and are testable without the heavy libraries installed.
"""
