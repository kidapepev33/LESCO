const CACHE_NAMESPACE = 'prisma-';
const CACHE_VERSION = 'ui-v2';
const CACHE_NAME = `${CACHE_NAMESPACE}${CACHE_VERSION}`;

const APP_SHELL = [
  '/',
  '/lesco-a-texto',
  '/texto-a-lesco',
  '/nosotros',
  '/ayuda',
  '/manifest.webmanifest',
  '/assets/css/main.css',
  '/assets/css/shared.css',
  '/assets/css/navigation.css',
  '/assets/css/pages/index.css',
  '/assets/css/pages/lesco-a-texto.css',
  '/assets/css/pages/texto-a-lesco.css',
  '/assets/css/pages/nosotros.css',
  '/assets/css/pages/ayuda.css',
  '/assets/css/pages/mobile-layout.css',
  '/assets/js/app.js',
  '/assets/images/IMAGOTIPO.png',
  '/assets/images/ISOTIPOPRISMA.png',
  '/assets/images/LOGOTIPOPRISMA.png',
  '/pwa/register.js',
  '/pwa/pwa.css',
  '/pwa/icons/icon-192.png',
  '/pwa/icons/icon-512.png'
];

const LIVE_PATHS = new Set([
  '/health',
  '/resultado',
  '/frame',
  '/stream',
  '/texto-a-lesco/solicitar',
  '/texto-a-lesco/stream'
]);

function isLiveRequest(request, url) {
  return request.method !== 'GET' || LIVE_PATHS.has(url.pathname);
}

async function replaceShellCache() {
  const temporaryName = `${CACHE_NAME}-refresh`;
  await caches.delete(temporaryName);
  const temporaryCache = await caches.open(temporaryName);

  try {
    await temporaryCache.addAll(APP_SHELL);
    await caches.delete(CACHE_NAME);
    const currentCache = await caches.open(CACHE_NAME);
    const requests = await temporaryCache.keys();
    await Promise.all(requests.map(async (request) => {
      const response = await temporaryCache.match(request);
      if (response) await currentCache.put(request, response);
    }));
  } finally {
    await caches.delete(temporaryName);
  }
}

self.addEventListener('install', (event) => {
  event.waitUntil(replaceShellCache().then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names
      .filter((name) => name.startsWith(CACHE_NAMESPACE) && name !== CACHE_NAME)
      .map((name) => caches.delete(name)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (url.origin !== self.location.origin || isLiveRequest(request, url)) return;

  event.respondWith((async () => {
    try {
      const response = await fetch(request);
      if (response.ok) {
        const cache = await caches.open(CACHE_NAME);
        await cache.put(request, response.clone());
      }
      return response;
    } catch (error) {
      const cached = await caches.match(request);
      if (cached) return cached;
      if (request.mode === 'navigate') {
        const fallback = await caches.match('/');
        if (fallback) return fallback;
      }
      throw error;
    }
  })());
});

self.addEventListener('message', (event) => {
  if (event.data?.type !== 'PRISMA_REFRESH_CACHE') return;

  event.waitUntil((async () => {
    try {
      await replaceShellCache();
      event.ports[0]?.postMessage({ ok: true, cache: CACHE_NAME });
    } catch (error) {
      event.ports[0]?.postMessage({ ok: false, error: String(error) });
    }
  })());
});
