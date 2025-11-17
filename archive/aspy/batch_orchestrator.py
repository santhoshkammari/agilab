import threading
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple
import copy

class BatchContext:
    """Context manager for recording calls during batch tracing."""

    def __init__(self):
        self.calls = []
        self.call_id_counter = 0
        self.results = {}

    def record_call(self, predict_module, inputs):
        """Record a call and return a placeholder result."""
        call_id = self.call_id_counter
        self.call_id_counter += 1

        call_record = {
            'id': call_id,
            'module': predict_module,
            'inputs': inputs,
            'dependencies': []
        }

        # Check inputs for dependencies on previous calls
        for key, value in inputs.items():
            if isinstance(value, CallPlaceholder):
                call_record['dependencies'].append(value.call_id)

        self.calls.append(call_record)

        # Return placeholder for this call
        return CallPlaceholder(call_id)

class CallPlaceholder:
    """Placeholder for a call result during tracing."""

    def __init__(self, call_id):
        self.call_id = call_id

    def __getattr__(self, name):
        # Return another placeholder for attribute access
        return CallPlaceholder(f"{self.call_id}.{name}")

class BatchOrchestrator:
    """Orchestrates batch execution with dependency-aware optimization."""

    def __init__(self, module):
        self.module = module

    def batch_execute(self, inputs_list):
        """Execute module on a batch of inputs with optimal batching."""
        if not inputs_list:
            return []

        # Step 1: Trace execution to build dependency graph
        execution_plan = self._build_execution_plan(inputs_list[0])

        # Step 2: Execute in dependency levels
        batch_size = len(inputs_list)
        all_results = [{}] * batch_size  # Results for each input

        for level in execution_plan:
            self._execute_level(level, inputs_list, all_results)

        # Step 3: Extract final results
        final_results = []
        for i in range(batch_size):
            # Get the final result (last call in the trace)
            final_call_id = max(all_results[i].keys()) if all_results[i] else 0
            final_results.append(all_results[i].get(final_call_id))

        return final_results

    def _build_execution_plan(self, sample_input):
        """Trace execution to build dependency graph and group by levels."""
        # Create batch context for tracing
        batch_context = BatchContext()

        # Patch all Predict modules to use batch context
        self._patch_predict_modules(batch_context)

        try:
            # Trace execution with sample input
            self.module.forward(**sample_input)

            # Build dependency levels
            levels = self._build_dependency_levels(batch_context.calls)

        finally:
            # Restore original modules
            self._unpatch_predict_modules()

        return levels

    def _patch_predict_modules(self, batch_context):
        """Patch all Predict modules to record calls instead of executing."""
        from .predict.predict import Predict

        for attr_name in dir(self.module):
            attr = getattr(self.module, attr_name)
            if isinstance(attr, Predict):
                # Store original forward method
                attr._original_forward = attr.forward
                # Replace with recording version
                attr.forward = lambda **inputs, module=attr: batch_context.record_call(module, inputs)
                attr.batch_context = batch_context

    def _unpatch_predict_modules(self):
        """Restore original forward methods."""
        from .predict.predict import Predict

        for attr_name in dir(self.module):
            attr = getattr(self.module, attr_name)
            if isinstance(attr, Predict) and hasattr(attr, '_original_forward'):
                attr.forward = attr._original_forward
                delattr(attr, '_original_forward')
                if hasattr(attr, 'batch_context'):
                    delattr(attr, 'batch_context')

    def _build_dependency_levels(self, calls):
        """Group calls into dependency levels for optimal batching."""
        # Build dependency graph
        dependencies = {}
        for call in calls:
            dependencies[call['id']] = call['dependencies']

        # Topological sort to find levels
        levels = []
        remaining_calls = {call['id']: call for call in calls}

        while remaining_calls:
            # Find calls with no dependencies on remaining calls
            current_level = []
            for call_id, call in list(remaining_calls.items()):
                deps_in_remaining = [dep for dep in call['dependencies'] if dep in remaining_calls]
                if not deps_in_remaining:
                    current_level.append(call)
                    del remaining_calls[call_id]

            if not current_level:
                # Circular dependency - shouldn't happen in practice
                break

            levels.append(current_level)

        return levels

    def _execute_level(self, level_calls, inputs_list, all_results):
        """Execute all calls in a dependency level efficiently."""
        if not level_calls:
            return

        # Group calls by module for batching
        calls_by_module = defaultdict(list)
        for call in level_calls:
            calls_by_module[call['module']].append(call)

        # Execute each module's calls in batch
        for module, calls in calls_by_module.items():
            self._execute_module_batch(module, calls, inputs_list, all_results)

    def _execute_module_batch(self, module, calls, inputs_list, all_results):
        """Execute a batch of calls for a single module."""
        batch_size = len(inputs_list)

        # Prepare batch inputs
        batch_inputs = []
        call_mapping = []  # Maps batch index to (input_index, call)

        for input_idx in range(batch_size):
            for call in calls:
                # Resolve dependencies for this input
                resolved_inputs = {}
                for key, value in call['inputs'].items():
                    if isinstance(value, CallPlaceholder):
                        # Look up result from previous calls
                        resolved_inputs[key] = all_results[input_idx].get(value.call_id)
                    else:
                        resolved_inputs[key] = value

                batch_inputs.append(resolved_inputs)
                call_mapping.append((input_idx, call))

        # Execute batch
        if batch_inputs:
            batch_results = module.batch_forward(batch_inputs)

            # Store results
            for i, result in enumerate(batch_results):
                input_idx, call = call_mapping[i]
                all_results[input_idx][call['id']] = result