"""
Main RAG Script - Standalone Query Processor

This script demonstrates how to use the RAG system independently
of the API. Useful for testing, debugging, and batch processing.
"""

import json
import sys
import argparse
from typing import Dict, Any
import logging

from model_selection import llm
from helper import context_outputer
from prompts import prompt_2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGProcessor:
    """RAG query processor with context retrieval and LLM generation."""
    
    def __init__(self):
        """Initialize the RAG processor with necessary components."""
        self.llm = llm
        self.prompt = prompt_2
        logger.info("RAG Processor initialized")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a single query through the RAG pipeline.
        
        Args:
            query: User's question or query
            
        Returns:
            Dictionary containing:
                - answer: Generated response
                - page_number: Source page reference
                - query: Original query
                
        Raises:
            ValueError: If LLM response cannot be parsed
            KeyError: If required fields are missing
        """
        try:
            logger.info(f"Processing query: '{query}'")
            
            # Step 1: Retrieve relevant context
            logger.info("Retrieving context...")
            context_final = context_outputer(query)
            
            # Step 2: Create chain and invoke LLM
            logger.info("Invoking LLM...")
            chain = self.prompt | self.llm
            result = chain.invoke({
                "query": query,
                "context": context_final
            })
            
            # Step 3: Parse and validate response
            logger.info("Parsing response...")
            output_str = result.content
            output_dict = json.loads(output_str)
            
            # Validate required fields
            if "answer" not in output_dict:
                raise KeyError("Response missing 'answer' field")
            if "page_number" not in output_dict:
                raise KeyError("Response missing 'page_number' field")
            
            # Add original query to output
            output_dict["query"] = query
            
            logger.info(f"Query processed successfully. Source: Page {output_dict['page_number']}")
            return output_dict
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except KeyError as e:
            logger.error(f"Missing required field in response: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def process_batch(self, queries: list[str]) -> list[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of result dictionaries
        """
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            try:
                result = self.process_query(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process query '{query}': {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "answer": None,
                    "page_number": None
                })
        return results
    
    def display_result(self, result: Dict[str, Any]) -> None:
        """
        Display a formatted result.
        
        Args:
            result: Query result dictionary
        """
        print("\n" + "="*80)
        print(f"QUERY: {result.get('query', 'N/A')}")
        print("="*80)
        
        if "error" in result:
            print(f"\n❌ ERROR: {result['error']}")
        else:
            print(f"\n📄 SOURCE: Page {result.get('page_number', 'Unknown')}")
            print(f"\n💡 ANSWER:\n{result.get('answer', 'No answer')}")
        
        print("="*80 + "\n")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="RAG Query Processor - Process queries using RAG system"
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="Query to process (if not provided, uses default)"
    )
    parser.add_argument(
        "-b", "--batch",
        type=str,
        help="Path to file containing queries (one per line)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize processor
    processor = RAGProcessor()
    
    # Process batch file
    if args.batch:
        try:
            with open(args.batch, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Processing {len(queries)} queries from batch file")
            results = processor.process_batch(queries)
            
            # Display results
            for result in results:
                processor.display_result(result)
            
            # Save to output file if specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {args.output}")
            
        except FileNotFoundError:
            logger.error(f"Batch file not found: {args.batch}")
            sys.exit(1)
    
    # Process single query
    else:
        # Use provided query or default
        query = args.query or "explain Preamble in bullet points"
        
        try:
            result = processor.process_query(query)
            processor.display_result(result)
            
            # Save to output file if specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Result saved to {args.output}")
                
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()