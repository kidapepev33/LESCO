const menuButton = document.querySelector('.menu-button');
const menuBackdrop = document.querySelector('.menu-backdrop');

function closeMenu() {
  document.body.classList.remove('menu-open');
  menuButton?.setAttribute('aria-expanded', 'false');
  menuButton?.setAttribute('aria-label', 'Abrir menú');
}

menuButton?.addEventListener('click', () => {
  const open = !document.body.classList.contains('menu-open');
  document.body.classList.toggle('menu-open', open);
  menuButton.setAttribute('aria-expanded', String(open));
  menuButton.setAttribute('aria-label', open ? 'Cerrar menú' : 'Abrir menú');
});
menuBackdrop?.addEventListener('click', closeMenu);
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeMenu(); });

const sign = document.querySelector('#sign');
if (sign) {
  const confidence = document.querySelector('#confidence');
  const status = document.querySelector('#status');
  const connection = document.querySelector('#connection');
  const cameraFeed = document.querySelector('#camera-feed');
  const cameraEmpty = document.querySelector('#camera-empty');

  async function updateResult() {
    try {
      const response = await fetch('/resultado', { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const result = await response.json();
      sign.textContent = result['seña'] || '—';
      confidence.textContent = result.confianza === null || result.confianza === undefined
        ? (result.estado === 'WAITING' ? 'Esperando seña' : 'Esperando una seña…')
        : `Confianza: ${(result.confianza * 100).toFixed(1)}%`;
      status.textContent = 'Conectado';
      connection.dataset.state = 'connected';
    } catch (error) {
      status.textContent = 'Sin conexión con la API local';
      connection.dataset.state = 'offline';
    }
  }

  if (cameraFeed) {
    cameraFeed.addEventListener('load', () => {
      cameraFeed.hidden = false;
      cameraEmpty.hidden = true;
    });
    cameraFeed.addEventListener('error', () => {
      cameraFeed.hidden = true;
      cameraEmpty.hidden = false;
    });
  }

  updateResult();
  setInterval(updateResult, 500);
}

const translationForm = document.querySelector('#translation-form');
if (translationForm) {
  const textInput = document.querySelector('#translation-text');
  const translationStatus = document.querySelector('#translation-status');
  const submitButton = translationForm.querySelector('button[type="submit"]');
  const signVideo = document.querySelector('#sign-video');

  signVideo?.addEventListener('load', () => {
    signVideo.closest('.sign-video-frame')?.classList.add('has-frame');
  });

  translationForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const text = textInput.value.trim();
    if (!text) {
      translationStatus.textContent = 'Escribe una seña para reproducir.';
      textInput.focus();
      return;
    }

    submitButton.disabled = true;
    translationStatus.textContent = 'Buscando video…';
    try {
      const response = await fetch('/texto-a-lesco/solicitar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texto: text }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      translationStatus.textContent = `Reproduciendo: ${result.texto}`;
    } catch (error) {
      translationStatus.textContent = error.message || 'No se pudo solicitar el video.';
    } finally {
      submitButton.disabled = false;
    }
  });
}
