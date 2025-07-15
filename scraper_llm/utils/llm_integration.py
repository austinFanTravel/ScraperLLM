"""Local LLM integration using llama-cpp-python."""
from typing import List, Optional
from pathlib import Path
from llama_cpp import Llama
import logging

logger = logging.getLogger(__name__)

class LocalLLM:
    def __init__(
        self,
        model_path: str = "models/mistral-7b-instruct.Q4_K_M.gguf",
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,  # -1 for all layers
        n_threads: Optional[int] = None,
    ):
        """Initialize the local LLM.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_threads: Number of CPU threads to use (None for auto)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        logger.info(f"Loading model from {model_path}...")
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
        )
        logger.info("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (lower = more deterministic)
            top_p: Nucleus sampling parameter
            stop: List of strings to stop generation at
            
        Returns:
            Generated text
        """
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo=False,
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def answer_question(self, context: str, question: str) -> str:
        """Generate an answer to a question based on the given context.
        
        Args:
            context: The context to use for answering
            question: The question to answer
            
        Returns:
            The generated answer
        """
        prompt = f"""<s>[INST] Answer the question based on the context below. Keep the answer concise and accurate.
        
Context: {context}

Question: {question}

Answer: [/INST]"""
        
        return self.generate(
            prompt,
            max_tokens=512,
            temperature=0.3,  # Lower temperature for more focused answers
            stop=["</s>", "[INST]"],
        )
