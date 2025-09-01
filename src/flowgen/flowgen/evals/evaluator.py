import os
import uuid
import hashlib
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

def custom_accuracy(prompt, answer, llm):
    """Simple accuracy metric based on answer content"""
    return 1.0 if "correct" in answer.lower() else 0.0

def llm_judge(prompt, answer, llm):
    """Use LLM to judge answer quality on a scale of 1-10"""
    judge_prompt = f"Rate this answer on a scale of 1-10:\nQuestion: {prompt}\nAnswer: {answer}\nRating:"
    result = llm(judge_prompt)
    content = result.get('content', '5')
    try:
        return float(content.strip().split()[0])
    except:
        return 5.0

def bleu_score(prompt, answer, llm):
    """Simple BLEU-like score (placeholder implementation)"""
    # In real implementation, you'd use proper BLEU calculation
    words_answer = set(answer.lower().split())
    words_prompt = set(prompt.lower().split())
    if not words_answer:
        return 0.0
    overlap = len(words_answer.intersection(words_prompt))
    return overlap / len(words_answer)

# Default metrics registry - supports both function objects and string keys
METRICS = {
    "accuracy": custom_accuracy,
    "llm_judge": llm_judge,
    "bleu": bleu_score,
}

class Evaluator:
    def __init__(self, metrics=None, llm=None,eval_name=None, cache_dir=".cache_evals"):
        self.metrics = metrics or ["accuracy"]
        self.llm = llm
        self.eval_name = eval_name or f"eval_{str(uuid.uuid4())[:8]}"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.eval_cache_dir = self.cache_dir / self.eval_name
        self.eval_cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, dataset_name_or_path, metric_name):
        """Generate cache key for dataset + metric combination"""
        cache_input = f"{dataset_name_or_path}_{metric_name}_{self.eval_name}"
        return hashlib.md5(cache_input.encode()).hexdigest()[:12]
    
    def _load_from_cache(self, cache_key):
        """Load cached results if available"""
        cache_file = self.eval_cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return None
    
    def _save_to_cache(self, data, cache_key, batch_idx=None):
        """Save batch results to cache"""
        if batch_idx is not None:
            cache_file = self.eval_cache_dir / f"{cache_key}_batch_{batch_idx:04d}.parquet"
        else:
            cache_file = self.eval_cache_dir / f"{cache_key}.parquet"
        
        df = pd.DataFrame(data)
        df.to_parquet(cache_file, index=False)
        return cache_file
    
    def __call__(self, dataset_name_or_path, cache=True, batch_size=4):
        print(f"🚀 Starting evaluation: {self.eval_name}")
        print(f"📁 Cache directory: {self.eval_cache_dir}")
        
        dataset = load_dataset(dataset_name_or_path)["test"]
        
        with tqdm(total=len(self.metrics), desc="📊 Processing metrics") as metric_pbar:
            for metric_name in self.metrics:
                # Support both string keys and function objects directly
                if isinstance(metric_name, str):
                    metric_fn = METRICS[metric_name]
                    result_key = metric_name
                else:
                    # Direct function object
                    metric_fn = metric_name
                    result_key = metric_name.__name__
                
                cache_key = self._get_cache_key(dataset_name_or_path, result_key)
                
                # Check cache first
                if cache:
                    cached_data = self._load_from_cache(cache_key)
                    if cached_data is not None:
                        print(f"📋 Loading {result_key} from cache...")
                        # Add cached results back to dataset
                        dataset = dataset.add_column(result_key, cached_data[result_key].tolist())
                        metric_pbar.update(1)
                        continue
                
                print(f"🔄 Computing {result_key}...")
                
                # Process with progress bar and batch saving
                def process_batch_with_cache(batch, indices):
                    batch_results = []
                    with tqdm(total=len(batch["prompt"]), desc=f"  Batch processing", leave=False) as batch_pbar:
                        for prompt, answer in zip(batch["prompt"], batch["answer"]):
                            result = metric_fn(prompt, answer, self.llm)
                            batch_results.append(result)
                            batch_pbar.update(1)
                    
                    # Save batch to cache
                    batch_data = {
                        "prompt": batch["prompt"],
                        "answer": batch["answer"], 
                        result_key: batch_results,
                        "indices": indices
                    }
                    
                    batch_idx = indices[0] // batch_size
                    self._save_to_cache(batch_data, cache_key, batch_idx)
                    
                    return {result_key: batch_results}
                
                dataset = dataset.map(
                    process_batch_with_cache,
                    batched=True,
                    batch_size=batch_size,
                    with_indices=True,
                    desc=f"🎯 {result_key}"
                )
                
                # Save final metric results
                final_data = {
                    "prompt": dataset["prompt"],
                    "answer": dataset["answer"],
                    result_key: dataset[result_key]
                }
                self._save_to_cache(final_data, cache_key)
                
                metric_pbar.update(1)
        
        print(f"✅ Evaluation complete! Results saved in: {self.eval_cache_dir}")
        return dataset
