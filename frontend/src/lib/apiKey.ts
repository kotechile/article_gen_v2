// Single source of truth for the backend's X-API-Key gate.
//
// The value is supplied at build time by VITE_API_KEY and baked into the bundle,
// so it is a bot filter, not a secret: it must never grant anything the UI itself
// cannot already do. Callers must not hardcode their own copy — every request that
// bypasses this constant keeps working right up until the key rotates, and then
// fails on its own with a 401 nobody can trace back to this file.
export const API_KEY: string = (import.meta.env.VITE_API_KEY as string | undefined) ?? '';

if (!API_KEY) {
    // Loud but non-fatal: requests still go out and the backend answers 401, which
    // is easier to diagnose than a bundle that refuses to mount.
    console.error(
        '[apiKey] VITE_API_KEY was not set at build time — every backend request will be rejected with 401. ' +
        'Set it as a build-time variable in the deployment environment and rebuild.'
    );
}
