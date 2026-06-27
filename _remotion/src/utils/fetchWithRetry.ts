import { useEffect, useState } from "react";
import { delayRender, continueRender, staticFile } from "remotion";

const assetCache: Record<string, string> = {};

/**
 * Resolves local assets using Remotion's staticFile utility, or leaves external URLs as is.
 */
export function resolveAssetUrl(url: string): string {
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://") || url.startsWith("file://")) {
    return url;
  }
  // It's a relative path to public folder, wrap in staticFile
  return staticFile(url);
}

/**
 * Fetches an external asset with up to 3 download retries and exponential backoff.
 * Converts the resolved blob to a local Object URL.
 */
export async function fetchAssetWithRetry(
  url: string,
  retries = 3,
  backoff = 1000
): Promise<string> {
  if (assetCache[url]) {
    return assetCache[url];
  }

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const blob = await response.blob();
      const localUrl = URL.createObjectURL(blob);
      assetCache[url] = localUrl;
      return localUrl;
    } catch (e) {
      if (attempt === retries) {
        throw new Error(
          `Failed to load asset ${url} after ${retries} attempts. Error: ${
            (e as Error).message
          }`
        );
      }
      console.warn(
        `Attempt ${attempt} to download ${url} failed. Retrying in ${backoff}ms...`
      );
      await new Promise((resolve) => setTimeout(resolve, backoff));
      backoff *= 2; // exponential backoff
    }
  }
  throw new Error("Unreachable");
}

/**
 * React hook that preloads an asset.
 * Local assets (staticFile / file://) bypass fetch completely.
 * Remote URLs (http/https) are downloaded with retry logic.
 */
export function useResilientAsset(url: string | undefined) {
  const [localUrl, setLocalUrl] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!url) {
      setLocalUrl(null);
      return;
    }

    const resolved = resolveAssetUrl(url);

    // Bypass fetch for file:// and relative local files
    if (resolved.startsWith("file://") || !resolved.startsWith("http")) {
      setLocalUrl(resolved);
      return;
    }

    // Check cache
    if (assetCache[resolved]) {
      setLocalUrl(assetCache[resolved]);
      return;
    }

    const handle = delayRender(`Downloading resilient asset: ${resolved}`);

    fetchAssetWithRetry(resolved)
      .then((resolvedUrl) => {
        setLocalUrl(resolvedUrl);
        continueRender(handle);
      })
      .catch((err) => {
        setError(err);
        continueRender(handle);
        console.error(`FATAL: Asset download failed: ${resolved}`);
        setTimeout(() => {
          throw err;
        }, 0);
      });
  }, [url]);

  return { localUrl, error };
}
export default useResilientAsset;
