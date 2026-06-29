/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Enables the dev-only observability Console. Off unless set to "1". */
  readonly VITE_DEBUG_PANELS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
