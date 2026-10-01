(() => {
  if (!('serviceWorker' in navigator)) return;

  const controls = document.createElement('div');
  controls.className = 'pwa-controls';
  controls.innerHTML = `
    <button class="pwa-refresh" type="button">Actualizar recursos guardados</button>
    <span class="pwa-status" role="status" aria-live="polite"></span>
  `;
  document.body.append(controls);

  const button = controls.querySelector('.pwa-refresh');
  const status = controls.querySelector('.pwa-status');

  navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch(() => {
    status.textContent = 'Modo sin conexión no disponible.';
    button.disabled = true;
  });

  button.addEventListener('click', async () => {
    button.disabled = true;
    status.textContent = 'Actualizando…';

    try {
      const registration = await navigator.serviceWorker.ready;
      const worker = registration.active || registration.waiting || registration.installing;
      if (!worker) throw new Error('No hay un service worker activo.');

      const channel = new MessageChannel();
      const result = new Promise((resolve, reject) => {
        channel.port1.onmessage = (event) => {
          event.data?.ok ? resolve(event.data) : reject(new Error(event.data?.error || 'Error desconocido.'));
        };
      });

      worker.postMessage({ type: 'PRISMA_REFRESH_CACHE' }, [channel.port2]);
      await result;
      status.textContent = 'Recursos actualizados.';
    } catch (error) {
      status.textContent = 'No se pudieron actualizar los recursos.';
    } finally {
      button.disabled = false;
    }
  });
})();
