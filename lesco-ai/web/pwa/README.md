# PWA local de Prisma

Esta carpeta contiene únicamente la capa PWA. La interfaz, Flask y los flujos
de reconocimiento siguen viviendo en sus ubicaciones originales.

## Identidad

- Nombre, colores, `start_url` e iconos: `manifest.webmanifest`.
- Logo base: `web/assets/images/Logopwa.png` (respeta mayúsculas).
- Iconos generados: `icons/icon-192.png` y `icons/icon-512.png`.

Para reemplazar el logo, sustituye `web/assets/images/Logopwa.png` y vuelve a
generar ambos iconos cuadrados:

```bash
.venv/bin/python web/pwa/generate_icons.py
```

El manifest usa rutas servidas por Flask.

## Caché

`sw.js` usa el prefijo `prisma-` y una versión explícita (`CACHE_VERSION`).
Al cambiar recursos incompatibles, incrementa esa versión. Durante la
activación solo se eliminan cachés cuyo nombre comienza con `prisma-`.

La interfaz usa red primero y caché como respaldo. Las detecciones, streams,
frames, solicitudes de traducción y videos siempre van directamente a la red y
nunca se reutilizan desde caché.

La acción “Actualizar recursos guardados” solicita manualmente la reconstrucción
del respaldo. No hay temporizadores ni comprobaciones periódicas adicionales.

## Rutas Flask

- `/manifest.webmanifest`: manifest compartido.
- `/sw.js`: worker servido desde raíz para controlar todo Prisma.
- `/pwa/...`: registro, estilos e iconos.

La instalación real la realiza el navegador. Abrir Chrome con `--app=URL` crea
una ventana sin controles, pero no demuestra que la PWA esté instalada.
