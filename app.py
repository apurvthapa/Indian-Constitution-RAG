from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any
import json
import logging

# Import RAG components
from model_selection import llm
from helper import context_outputer
from prompts import prompt_2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
PDF_FILE = BASE_DIR / "constitution raw pdf" / "constitution-better.pdf"

# Initialize FastAPI app
app = FastAPI(
    title="Centralised RAG API",
    description="Retrieval-Augmented Generation API for document queries",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Models
class QueryRequest(BaseModel):
    query: str = Field(
        ..., 
        description="User's question or query",
        min_length=1,
        example="explain Preamble in bullet points"
    )


# Response Models
class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer to the query")
    page_number: str | int = Field(..., description="Source page number")
    query: str = Field(..., description="Original query")


class HealthResponse(BaseModel):
    status: str
    message: str


class ErrorResponse(BaseModel):
    detail: str
    error_type: str


# RAG Processing Function
def process_rag_query(query: str) -> Dict[str, Any]:
    """
    Process a RAG query and return the answer with metadata.
    
    Args:
        query: User's question
        
    Returns:
        Dictionary with answer and page_number
        
    Raises:
        ValueError: If LLM response cannot be parsed
        KeyError: If expected keys are missing from response
    """
    try:
        logger.info(f"Processing query: {query}")
        
        # Step 1: Get relevant context
        context_final = context_outputer(query)
        logger.info("Context retrieved successfully")
        
        # Step 2: Create and invoke the chain
        chain = prompt_2 | llm
        result = chain.invoke({
            "query": query,
            "context": context_final
        })
        logger.info("LLM invoked successfully")
        
        # Step 3: Parse the result
        output_str = result.content
        output_dict = json.loads(output_str)
        
        # Step 4: Validate required keys
        if "answer" not in output_dict or "page_number" not in output_dict:
            raise KeyError("Response missing 'answer' or 'page_number' field")
        
        logger.info(f"Query processed successfully. Page: {output_dict['page_number']}")
        return output_dict
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")
    except KeyError as e:
        logger.error(f"Missing key in response: {str(e)}")
        raise KeyError(f"LLM response missing expected field: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in RAG processing: {str(e)}")
        raise


# API Endpoints
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is running"
)
async def health_check():
    """Health check endpoint to verify API is operational."""
    return HealthResponse(
        status="healthy",
        message="RAG API is running"
    )


@app.post(
    "/rag",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query RAG System",
    description="Submit a query and receive an AI-generated answer with source reference",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "answer": "The Preamble is the introduction to the Constitution...",
                        "page_number": "1",
                        "query": "explain Preamble in bullet points"
                    }
                }
            }
        },
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"}
    }
)
async def query_rag(request: QueryRequest):
    """
    Process a query through the RAG system.
    
    - **query**: Your question or search query
    
    Returns the answer along with the source page number.
    """
    try:
        # Process the query
        result = process_rag_query(request.query)
        
        # Return structured response
        return QueryResponse(
            answer=result["answer"],
            page_number=result["page_number"],
            query=request.query
        )
        
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except KeyError as e:
        logger.error(f"Key error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your query: {str(e)}"
        )


@app.get(
    "/",
    summary="Root",
    description="Frontend entrypoint"
)
async def root():
    """Serve the frontend."""
    return FileResponse(INDEX_FILE)


@app.get(
    "/constitution-pdf",
    summary="Preloaded Constitution PDF",
    description="Serves the bundled Constitution PDF for the frontend viewer"
)
async def constitution_pdf():
    """Serve the preloaded Constitution PDF."""
    return FileResponse(PDF_FILE, media_type="application/pdf", filename=PDF_FILE.name)


@app.get(
    "/api",
    summary="API information",
    description="Basic metadata about the available API endpoints"
)
async def api_info():
    """API information endpoint."""
    return {
        "message": "Welcome to the Centralised RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "rag_endpoint": "/rag",
        "frontend": "/",
        "pdf": "/constitution-pdf"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
