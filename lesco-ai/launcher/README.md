# Lanzador local de Prisma

Esta carpeta no contiene reconocimiento ni lógica web. Su única función es
comprobar el backend, reutilizar `prisma.py` y abrir Prisma.

## Configuración

Edita `config.json`:

- `app_name`: nombre mostrado por el lanzador.
- `base_url`: dirección Flask local; inicialmente `http://127.0.0.1:5000`.
- `auto_start_backend`: inicia `prisma.py` si Prisma no responde.
- `stop_backend_on_exit`: si se activa, espera al cierre real de la ventana y
  solo cierra el backend que este mismo lanzador haya iniciado.
- `backend_start_timeout_seconds`: espera máxima por `/health`.
- `fullscreen`: abre la aplicación en pantalla completa real con
  `--start-fullscreen`.
- `browser_profile_dir`: perfil dedicado de Chrome. Vacío usa el directorio de
  estado local de Prisma; una ruta relativa se resuelve desde el proyecto.
- `browser_candidates`: navegadores Linux aceptados.
- `installed_app_id`: ID de una PWA instalada en Chrome. Déjalo vacío para usar
  `--app=URL` como respaldo.

La ventana `--app=URL` elimina controles del navegador, pero no equivale ni se
presenta como prueba de instalación PWA. Para abrir la PWA realmente instalada,
configura su ID de Chrome en `installed_app_id`.

Chrome se abre con un perfil dedicado y un endpoint DevTools exclusivamente
local. El lanzador observa allí la ventana real de Prisma, incluso cuando el
comando de apertura se entrega a una instancia ya activa de ese perfil. Esto
evita asociar el cierre de Prisma con otras ventanas del perfil normal de Chrome.

## Ejecutar

```bash
./launcher/run_prisma.sh
```

El lanzador valida `/health`; por eso no confunde otro servicio que use el
puerto 5000 con Prisma. Un bloqueo temporal evita arranques duplicados cuando se
abren dos accesos a la vez.

Si `auto_start_backend` es `false`, no se ejecuta Python. Se abre la interfaz en
modo aplicación para permitir usar el respaldo offline, y los controles en vivo
indican que el backend no está disponible.

## Instalar el acceso directo en Linux

```bash
.venv/bin/python launcher/install_shortcut.py
```

El instalador calcula la ruta del checkout actual y crea
`~/.local/share/applications/prisma.desktop`. No hay rutas de la computadora del
desarrollador guardadas en el repositorio.

## Diagnóstico

Cuando el lanzador inicia el backend, su salida queda en
`launcher/logs/backend.log`. Esa carpeta se crea en tiempo de ejecución.
