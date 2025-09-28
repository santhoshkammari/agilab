from collections import defaultdict

from pydantic import BaseModel
from tqdm import tqdm

import aspy as a
lm = a.LM(api_base="http://192.168.170.76:8000")
a.configure(lm=lm)

class Extract(BaseModel):
    including: list
    excluding: list
    general: list

class EntityExtractor(a.Module):
    def __init__(self):
        super().__init__()
        self.extract = a.Predict(
            a.Signature("text -> Extract",
                       instructions='Extract ALL entities mentioned in the text. For "including": extract entities specifically mentioned as included. For "excluding": extract entities specifically mentioned as excluded. For "general": extract main/general entities that are neither explicitly included nor excluded. Example: "All Scheduled Commercial Banks (including Small Finance Banks and excluding RRBs)" should extract including: ["Small Finance Banks"], excluding: ["RRBs"], general: ["Scheduled Commercial Banks"]')
        )
        self.verify_including = a.Predict(
            a.Signature('text, entities:list -> verified_entities:list',
                       instructions="verify which entities are actually mentioned as specifically included in the text, return only those that exist")
        )
        self.verify_excluding = a.Predict(
            a.Signature('text, entities:list -> verified_entities:list',
                       instructions="verify which entities are actually mentioned as excluded in the text, return only those that are excluded")
        )
        self.verify_general = a.Predict(
            a.Signature('text, entities:list -> verified_entities:list',
                       instructions="verify which entities are mentioned as general/main entities (not specifically included or excluded) in the text, return only those that exist")
        )
        self.final_verifier = a.Predict(
            a.Signature('text, including:list, excluding:list, general:list -> including:list, excluding:list, general:list',
                       instructions="Final verification: Review the complete text and fix any missed entities or incorrect categorizations. Ensure ALL entities mentioned in the text are captured in the appropriate category. Look for patterns like 'including X', 'excluding Y', and main entity names.")
        )


    def _clean_entities(self, entities):
        """Remove 'all'/'All'/'ALL' from entity lists"""
        if not entities:
            return []
        cleaned = []
        for entity in entities:
            if isinstance(entity, str):
                if 'all' in entity.lower():
                    splits = entity.split()
                    x = []
                    for item in splits:
                        if item.lower().strip()!='all':
                            x.append(item)
                    cleaned.append(" ".join(x))
                else:
                    cleaned.append(entity)
            else:
                cleaned.append(entity)
        return cleaned

    def forward(self, text):
        # Step 1: Extract initial entities
        result = self.extract(text=text)

        # Clean extracted entities
        cleaned_including = self._clean_entities(result.including)
        cleaned_excluding = self._clean_entities(result.excluding)
        cleaned_general = self._clean_entities(result.general)

        # Step 2: Verify including entities are present in text
        if cleaned_including:
            verified_including = self.verify_including(text=text, entities=cleaned_including)
            final_including = self._clean_entities(verified_including.verified_entities)
        else:
            final_including = []

        # Step 3: Verify excluding entities are actually excluded in text
        if cleaned_excluding:
            verified_excluding = self.verify_excluding(text=text, entities=cleaned_excluding)
            final_excluding = self._clean_entities(verified_excluding.verified_entities)
        else:
            final_excluding = []

        # Step 4: Verify general entities are present in text
        if cleaned_general:
            verified_general = self.verify_general(text=text, entities=cleaned_general)
            final_general = self._clean_entities(verified_general.verified_entities)
        else:
            final_general = []

        # Step 5: Final verification to catch any missed entities
        final_result = self.final_verifier(
            text=text,
            including=final_including,
            excluding=final_excluding,
            general=final_general
        )

        return a.Prediction(
            including=self._clean_entities(final_result.including),
            excluding=self._clean_entities(final_result.excluding),
            general=self._clean_entities(final_result.general)
        )

    def batch_forward(self, texts):
        """Batch processing using the underlying LM batch capabilities"""
        # Step 1: Batch extract
        extract_inputs = [{"text": text} for text in texts]
        extract_results = self.extract.batch_forward(extract_inputs)

        # Step 2: Prepare batch verify inputs
        verify_including_inputs = []
        verify_excluding_inputs = []
        verify_general_inputs = []

        for i, (text, result) in enumerate(zip(texts, extract_results)):
            cleaned_including = self._clean_entities(result.including if hasattr(result, 'including') else [])
            cleaned_excluding = self._clean_entities(result.excluding if hasattr(result, 'excluding') else [])
            cleaned_general = self._clean_entities(result.general if hasattr(result, 'general') else [])

            verify_including_inputs.append({"text": text, "entities": cleaned_including} if cleaned_including else None)
            verify_excluding_inputs.append({"text": text, "entities": cleaned_excluding} if cleaned_excluding else None)
            verify_general_inputs.append({"text": text, "entities": cleaned_general} if cleaned_general else None)

        # Filter out None inputs for batch processing
        valid_including_inputs = [inp for inp in verify_including_inputs if inp is not None]
        valid_excluding_inputs = [inp for inp in verify_excluding_inputs if inp is not None]
        valid_general_inputs = [inp for inp in verify_general_inputs if inp is not None]

        # Batch verify all three types
        including_results = self.verify_including.batch_forward(valid_including_inputs) if valid_including_inputs else []
        excluding_results = self.verify_excluding.batch_forward(valid_excluding_inputs) if valid_excluding_inputs else []
        general_results = self.verify_general.batch_forward(valid_general_inputs) if valid_general_inputs else []

        # Reconstruct results
        final_results = []
        inc_idx = exc_idx = gen_idx = 0

        for i, text in enumerate(texts):
            if verify_including_inputs[i] is not None:
                final_including = self._clean_entities(including_results[inc_idx].verified_entities if inc_idx < len(including_results) else [])
                inc_idx += 1
            else:
                final_including = []

            if verify_excluding_inputs[i] is not None:
                final_excluding = self._clean_entities(excluding_results[exc_idx].verified_entities if exc_idx < len(excluding_results) else [])
                exc_idx += 1
            else:
                final_excluding = []

            if verify_general_inputs[i] is not None:
                final_general = self._clean_entities(general_results[gen_idx].verified_entities if gen_idx < len(general_results) else [])
                gen_idx += 1
            else:
                final_general = []

            final_results.append({
                "text": text,
                "including": final_including,
                "excluding": final_excluding,
                "general": final_general
            })

        # Step 4: Final verification batch
        final_verify_inputs = [
            {
                "text": result["text"],
                "including": result["including"],
                "excluding": result["excluding"],
                "general": result["general"]
            }
            for result in final_results
        ]

        final_verified_results = self.final_verifier.batch_forward(final_verify_inputs)

        # Step 5: Clean and return final results
        final_predictions = []
        for verified in final_verified_results:
            final_predictions.append(a.Prediction(
                including=self._clean_entities(verified.including if hasattr(verified, 'including') else []),
                excluding=self._clean_entities(verified.excluding if hasattr(verified, 'excluding') else []),
                general=self._clean_entities(verified.general if hasattr(verified, 'general') else [])
            ))

        return final_predictions

extractor = EntityExtractor()

import json
from pathlib import Path

data = json.loads(Path('/home/ntlpt59/master/products/rbi/metadata_fixing/metadata.json').read_text())
mt_map = defaultdict(list)
for k,v in data.items():
    for mt in v['meant_for']:
        mt_map[mt].append(k.replace(".json",""))

entities = {}

# Convert to list for batch processing
texts = list(mt_map.keys())
batch_size = 4 # Adjust based on your server capacity

for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
    batch_texts = texts[i:i+batch_size]

    # Batch process
    results = extractor.batch_forward(batch_texts)

    # Store results
    for text, result in zip(batch_texts, results):
        print(f"Text: {text}")
        print(f"Including: {result.including}")
        print(f"Excluding: {result.excluding}")
        print(f"General: {result.general}")
        print("---")
        entities[text] = {"including": result.including, "excluding": result.excluding, "general": result.general}

    # Save after each batch
    Path('res.json').write_text(json.dumps(entities, indent=2, ensure_ascii=False))
