from typing import Dict
from app.interfaces.recommendation_interfaces import IScoreCombiner

class ScoreCombiner(IScoreCombiner):
    def combine(
        self,
        content_scores: Dict[int, float],
        collaborative_scores: Dict[int, float],
        content_weight: float = 0.5,
        collaborative_weight: float = 0.5,
    ) -> Dict[int, float]:
        all_ids = set(content_scores) | set(collaborative_scores)
        return {
            vid: content_weight * content_scores.get(vid, 0.0)
                + collaborative_weight * collaborative_scores.get(vid, 0.0)
            for vid in all_ids
        }
