import time
import threading
import queue
import re
from kittentts import KittenTTS
import soundfile as sf
import subprocess
import os
from statistics import mean, median

class TTSBenchmark:
    def __init__(self):
        self.tts = KittenTTS("KittenML/kitten-tts-nano-0.1")
        self.audio_queue = queue.Queue()
        self.playing = False
        self.play_thread = None
        
        self.test_text = """
        The artificial intelligence revolution is transforming every aspect of human life. From healthcare diagnostics to autonomous vehicles, 
        machine learning algorithms are becoming increasingly sophisticated. Natural language processing enables computers to understand 
        and generate human-like text with remarkable accuracy. Computer vision systems can now identify objects, faces, and scenes 
        with superhuman precision. Deep learning networks, inspired by the human brain, consist of multiple layers that process 
        information hierarchically. These neural networks can learn complex patterns from vast amounts of data. Robotics and AI 
        are converging to create intelligent machines that can perform physical tasks. The future promises even more exciting 
        developments in artificial general intelligence. However, we must also consider the ethical implications of these powerful 
        technologies. Privacy, bias, and job displacement are important concerns that need careful consideration. As we advance 
        into this new era, collaboration between humans and machines will be crucial for success.
        """
    
    def chunk_by_sentences(self, text, num_sentences=2):
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), num_sentences):
            chunk = ' '.join(sentences[i:i+num_sentences])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def chunk_by_words(self, text, word_count=15):
        words = text.split()
        chunks = []
        for i in range(0, len(words), word_count):
            chunk = ' '.join(words[i:i+word_count])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def chunk_by_characters(self, text, char_count=100):
        chunks = []
        words = text.split()
        current_chunk = ""
        
        for word in words:
            if len(current_chunk + " " + word) <= char_count:
                current_chunk += " " + word if current_chunk else word
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    
    def generate_audio_chunk(self, text, chunk_id):
        start_time = time.time()
        audio = self.tts.generate(text, voice='expr-voice-2-f')
        generation_time = time.time() - start_time
        
        filename = f'chunk_{chunk_id}.wav'
        sf.write(filename, audio, 24000)
        
        return filename, generation_time, len(audio) / 24000
    
    def play_audio_file(self, filename):
        try:
            subprocess.run(['aplay', filename], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL, 
                         check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(['paplay', filename], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL, 
                             check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    subprocess.run(['afplay', filename], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL, 
                                 check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print(f"Could not play {filename}")
    
    def audio_player_worker(self):
        while self.playing:
            try:
                filename = self.audio_queue.get(timeout=1)
                if filename:
                    self.play_audio_file(filename)
                    os.remove(filename)
                    self.audio_queue.task_done()
            except queue.Empty:
                continue
    
    def benchmark_chunking_strategy(self, strategy_name, chunks):
        print(f"\n{'='*50}")
        print(f"Benchmarking: {strategy_name}")
        print(f"Number of chunks: {len(chunks)}")
        print(f"{'='*50}")
        
        generation_times = []
        audio_durations = []
        queue_wait_times = []
        
        self.playing = True
        self.play_thread = threading.Thread(target=self.audio_player_worker)
        self.play_thread.start()
        
        total_start = time.time()
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            
            filename, gen_time, duration = self.generate_audio_chunk(chunk, i)
            generation_times.append(gen_time)
            audio_durations.append(duration)
            
            queue_start = time.time()
            self.audio_queue.put(filename)
            queue_wait_times.append(time.time() - queue_start)
            
            print(f"  Generated in: {gen_time:.3f}s | Duration: {duration:.3f}s | Queue size: {self.audio_queue.qsize()}")
            
            if i < len(chunks) - 1:
                buffer_time = max(0, duration - gen_time) * 0.1
                time.sleep(buffer_time)
        
        self.audio_queue.join()
        self.playing = False
        if self.play_thread:
            self.play_thread.join()
        
        total_time = time.time() - total_start
        
        results = {
            'strategy': strategy_name,
            'chunks': len(chunks),
            'total_time': total_time,
            'avg_generation_time': mean(generation_times),
            'median_generation_time': median(generation_times),
            'avg_audio_duration': mean(audio_durations),
            'total_audio_duration': sum(audio_durations),
            'generation_times': generation_times,
            'audio_durations': audio_durations,
            'realtime_ratio': sum(audio_durations) / total_time
        }
        
        self.print_results(results)
        return results
    
    def print_results(self, results):
        print(f"\nResults for {results['strategy']}:")
        print(f"  Total chunks: {results['chunks']}")
        print(f"  Total time: {results['total_time']:.3f}s")
        print(f"  Total audio duration: {results['total_audio_duration']:.3f}s")
        print(f"  Average generation time: {results['avg_generation_time']:.3f}s")
        print(f"  Median generation time: {results['median_generation_time']:.3f}s")
        print(f"  Average chunk duration: {results['avg_audio_duration']:.3f}s")
        print(f"  Realtime ratio: {results['realtime_ratio']:.2f}x")
        print(f"  Efficiency: {'✓ Good' if results['realtime_ratio'] >= 1.0 else '✗ Needs optimization'}")
    
    def run_benchmark(self):
        print("TTS Chunking Strategy Benchmark")
        print("Testing optimal chunk sizes for continuous playback")
        
        strategies = [
            ("1 Sentence Chunks", self.chunk_by_sentences(self.test_text, 1)),
            ("2 Sentence Chunks", self.chunk_by_sentences(self.test_text, 2)),
            ("3 Sentence Chunks", self.chunk_by_sentences(self.test_text, 3)),
            ("10 Word Chunks", self.chunk_by_words(self.test_text, 10)),
            ("15 Word Chunks", self.chunk_by_words(self.test_text, 15)),
            ("20 Word Chunks", self.chunk_by_words(self.test_text, 20)),
            ("80 Character Chunks", self.chunk_by_characters(self.test_text, 80)),
            ("120 Character Chunks", self.chunk_by_characters(self.test_text, 120)),
            ("160 Character Chunks", self.chunk_by_characters(self.test_text, 160)),
        ]
        
        all_results = []
        
        for strategy_name, chunks in strategies:
            result = self.benchmark_chunking_strategy(strategy_name, chunks)
            all_results.append(result)
            time.sleep(2)
        
        self.generate_summary_report(all_results)
    
    def generate_summary_report(self, results):
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY REPORT")
        print(f"{'='*70}")
        
        print(f"{'Strategy':<25} {'Chunks':<8} {'Avg Gen':<8} {'Realtime':<10} {'Rating':<10}")
        print("-" * 70)
        
        best_strategy = max(results, key=lambda x: x['realtime_ratio'])
        
        for result in sorted(results, key=lambda x: x['realtime_ratio'], reverse=True):
            rating = "★★★★★" if result['realtime_ratio'] >= 1.5 else \
                    "★★★★☆" if result['realtime_ratio'] >= 1.2 else \
                    "★★★☆☆" if result['realtime_ratio'] >= 1.0 else \
                    "★★☆☆☆" if result['realtime_ratio'] >= 0.8 else "★☆☆☆☆"
            
            print(f"{result['strategy']:<25} {result['chunks']:<8} {result['avg_generation_time']:<8.3f} {result['realtime_ratio']:<10.2f}x {rating}")
        
        print(f"\n🏆 OPTIMAL STRATEGY: {best_strategy['strategy']}")
        print(f"   - {best_strategy['chunks']} chunks")
        print(f"   - {best_strategy['avg_generation_time']:.3f}s average generation time")
        print(f"   - {best_strategy['realtime_ratio']:.2f}x realtime performance")
        print(f"   - Recommended for continuous playback")

if __name__ == '__main__':
    benchmark = TTSBenchmark()
    benchmark.run_benchmark()