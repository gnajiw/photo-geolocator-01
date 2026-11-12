# Photo Geolocator Agent

An autonomous geolocation agent for make.fun's `photo-geolocator-01` competition.

## Strategy
1. **Parallel Stream Ingestion**: Queries `/photos` and processes batches concurrently.
2. **Vision-Guided Analysis**: Uses make.fun metered model gateway when API keys are configured, evaluating architectural cues, sun angle, driving orientation, and vegetation biomes.
3. **OpenStreetView-5M Geographic Anchoring**: Implements a calibrated geographic anchor distribution to ensure 0 unanswered photos, mitigating worst-case penalties.
4. **Deterministic Execution**: Pure functional mapping with deterministic sorting and replay guarantees.

## License
MIT
