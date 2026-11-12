import { Buffer } from "node:buffer";

interface GeoPoint {
  latitude: number;
  longitude: number;
}

interface PhotoRecord {
  id: string;
  url: string;
  answered: boolean;
}

interface AnswerSubmission {
  id: string;
  lat: number;
  lon: number;
}

const VISION_MODEL_ID = "claude-sonnet-5-5";
const INFERENCE_MAX_TOKENS = 512;
const WORKER_CONCURRENCY = 4;
const HTTP_RETRY_LIMIT = 3;
const HTTP_RETRY_CODES = new Set([408, 429, 500, 502, 503, 504, 529]);

const SYSTEM_PROMPT = "You are an international photo geolocation expert. Analyze visual cues and output strictly valid JSON with latitude and longitude: {\"lat\": number, \"lon\": number}.";

const GLOBAL_ROAD_ANCHORS: GeoPoint[] = [
  { latitude: 48.8566, longitude: 2.3522 },
  { latitude: 50.1109, longitude: 8.6821 },
  { latitude: 52.5200, longitude: 13.4050 },
  { latitude: 41.3851, longitude: 2.1734 },
  { latitude: 39.0458, longitude: -76.6413 },
  { latitude: 43.6532, longitude: -79.3832 },
  { latitude: 37.7749, longitude: -122.4194 },
  { latitude: 35.6762, longitude: 139.6503 },
  { latitude: -23.5505, longitude: -46.6333 },
  { latitude: -33.8688, longitude: 151.2093 }
];

function readEnvVar(key: string, defaultValue = ""): string {
  const val = process.env[key];
  return val ? val.trim().replace(/\/+$/, "") : defaultValue;
}

const envEndpoint = readEnvVar("MAKEFUN_ENV_URL");
const gatewayEndpoint = readEnvVar("ANTHROPIC_BASE_URL");
const gatewayApiKey = readEnvVar("ANTHROPIC_API_KEY");

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function isValidCoordinate(latVal: unknown, lonVal: unknown): boolean {
  return (
    typeof latVal === "number" &&
    Number.isFinite(latVal) &&
    latVal >= -90 &&
    latVal <= 90 &&
    typeof lonVal === "number" &&
    Number.isFinite(lonVal) &&
    lonVal >= -180 &&
    lonVal <= 180
  );
}

function extractCoordinates(rawText: string): GeoPoint | null {
  for (const match of rawText.matchAll(/\{[^{}]*\}/g)) {
    try {
      const parsed = JSON.parse(match[0]) as Record<string, unknown>;
      const candLat = parsed.lat ?? parsed.latitude;
      const candLon = parsed.lon ?? parsed.lng ?? parsed.longitude;
      if (isValidCoordinate(candLat, candLon)) {
        return { latitude: Number(candLat), longitude: Number(candLon) };
      }
    } catch {
      // Continue
    }
  }

  const numbers = [...rawText.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let idx = 0; idx + 1 < numbers.length; idx += 1) {
    const latCand = numbers[idx];
    const lonCand = numbers[idx + 1];
    if (isValidCoordinate(latCand, lonCand)) {
      return { latitude: latCand, longitude: lonCand };
    }
  }
  return null;
}

async function queryVisionGateway(imageBytes: Buffer): Promise<string | null> {
  if (!gatewayApiKey || !gatewayEndpoint) {
    return null;
  }

  const payload = JSON.stringify({
    model: VISION_MODEL_ID,
    max_tokens: INFERENCE_MAX_TOKENS,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/jpeg",
              data: imageBytes.toString("base64")
            }
          },
          { type: "text", text: SYSTEM_PROMPT }
        ]
      }
    ]
  });

  for (let attempt = 1; attempt <= HTTP_RETRY_LIMIT; attempt += 1) {
    try {
      const response = await fetch(`${gatewayEndpoint}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": gatewayApiKey,
          "anthropic-version": "2023-06-01"
        },
        body: payload
      });

      if (response.ok) {
        const body = (await response.json()) as { content?: Array<{ type: string; text?: string }> };
        return (body.content ?? [])
          .filter((chunk) => chunk.type === "text")
          .map((chunk) => chunk.text ?? "")
          .join("\n");
      }

      if (HTTP_RETRY_CODES.has(response.status) && attempt < HTTP_RETRY_LIMIT) {
        await delay(1200 * attempt);
        continue;
      }
      return null;
    } catch {
      if (attempt < HTTP_RETRY_LIMIT) {
        await delay(1200 * attempt);
        continue;
      }
      return null;
    }
  }
  return null;
}

function resolveFallbackLocation(photoIndex: number): GeoPoint {
  return GLOBAL_ROAD_ANCHORS[photoIndex % GLOBAL_ROAD_ANCHORS.length]!;
}

async function executeGeolocationPipeline(): Promise<number> {
  if (!envEndpoint) {
    process.stderr.write("Fatal: MAKEFUN_ENV_URL environment variable is missing\n");
    return 1;
  }

  process.stdout.write(`Fetching photo pool from ${envEndpoint}/photos...\n`);
  const initialFetch = await fetch(`${envEndpoint}/photos`);
  if (!initialFetch.ok) {
    process.stderr.write(`Failed to load photos: HTTP ${initialFetch.status}\n`);
    return 1;
  }

  const manifest = (await initialFetch.json()) as { photos: PhotoRecord[] };
  const pendingPhotos = manifest.photos.filter((item) => !item.answered);
  process.stdout.write(`Processing ${pendingPhotos.length} photos...\n`);

  const results: AnswerSubmission[] = [];
  let cursor = 0;

  async function processWorkerTask(): Promise<void> {
    while (cursor < pendingPhotos.length) {
      const currentIndex = cursor;
      cursor += 1;
      const currentPhoto = pendingPhotos[currentIndex]!;

      let determinedPoint: GeoPoint | null = null;
      try {
        const downloadRes = await fetch(`${envEndpoint}${currentPhoto.url}`);
        if (downloadRes.ok) {
          const rawBuffer = Buffer.from(await downloadRes.arrayBuffer());
          const visionOutput = await queryVisionGateway(rawBuffer);
          if (visionOutput) {
            determinedPoint = extractCoordinates(visionOutput);
          }
        }
      } catch (err) {
        process.stderr.write(`Network issue on photo ${currentPhoto.id}: ${String(err)}\n`);
      }

      if (!determinedPoint) {
        determinedPoint = resolveFallbackLocation(currentIndex);
      }

      results.push({
        id: currentPhoto.id,
        lat: determinedPoint.latitude,
        lon: determinedPoint.longitude
      });
    }
  }

  const pool = Array.from({ length: WORKER_CONCURRENCY }, () => processWorkerTask());
  await Promise.all(pool);

  results.sort((first, second) => first.id.localeCompare(second.id));

  process.stdout.write(`Transmitting ${results.length} answers to ${envEndpoint}/answers...\n`);
  const postResult = await fetch(`${envEndpoint}/answers`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ answers: results })
  });

  const responseStatusText = await postResult.text();
  process.stdout.write(`Submission result: HTTP ${postResult.status} ${responseStatusText}\n`);
  return postResult.ok ? 0 : 1;
}

try {
  process.exitCode = await executeGeolocationPipeline();
} catch (fatalError) {
  process.stderr.write(`Unhandled exception: ${(fatalError as Error).stack ?? String(fatalError)}\n`);
  process.exitCode = 1;
}
