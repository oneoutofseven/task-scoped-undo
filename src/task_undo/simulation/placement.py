"""Sample a supported object pose independently of its transport tilt."""


class StablePlacement:
    def _place_with_predicate(self, obj, predicate):
        # A reachable endpoint is not proof of a valid loaded-base path. Resample
        # the upstream support pose only while no action has been issued. Once
        # motion begins, errors retain the native handling and cleanup.
        from task_undo.simulation.interaction_search import pre_motion_planning_failure
        for attempt in range(3):
            stream = super()._place_with_predicate(obj, predicate)
            issued = False
            try:
                for action in stream:
                    issued = True
                    yield action
                return
            except Exception as error:
                if attempt == 2 or not pre_motion_planning_failure(error, issued):
                    raise
                self._record_interaction_search(
                    event="placement_resampled_before_motion",
                    placement_attempt=attempt + 1, error=str(error),
                    max_placement_attempts=3)
            finally:
                stream.close()

    def _sample_pose_with_object_and_predicate(self, predicate, held_obj, target_obj,
                                              world_aligned=False, **kwargs):
        from omnigibson.object_states import OnTop

        # The upstream world-aligned option preserves the object's current roll
        # and pitch. A transported mug can therefore be released on its side.
        # Use its canonical base-aligned bounding box on a sampled support pose.
        # Sampling, clearance, IK, navigation and release checks remain upstream.
        return super()._sample_pose_with_object_and_predicate(
            predicate, held_obj, target_obj,
            world_aligned=False if predicate is OnTop else world_aligned, **kwargs)
