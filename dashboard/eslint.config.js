import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      // Type-aware lint: catches the high-value class of bugs that
      // syntactic-only `recommended` misses (floating promises,
      // unnecessary conditionals, misused promises in JSX handlers).
      // We deliberately do NOT step up to `strictTypeChecked` — its
      // `no-unsafe-*` rules drown the run in noise around well-typed
      // boundaries (react-query, ReactFlow generics) without catching
      // additional real bugs in this codebase.
      tseslint.configs.recommendedTypeChecked,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // React Compiler is enabled in vite.config.ts and auto-memoizes
      // values whose inputs are stable. The `exhaustive-deps` rule's
      // entire purpose (enforce manual memoization correctness) is
      // subsumed; leaving it on produces noise around plain `const`
      // values that the compiler has already covered.
      'react-hooks/exhaustive-deps': 'off',
    },
  },
  // Tests use vi.fn().mockResolvedValue with arbitrary unknown bodies
  // — typing the fetch shim strictly would only add ceremony.
  {
    files: ['**/*.test.{ts,tsx}', 'src/test/**/*.{ts,tsx}'],
    rules: {
      '@typescript-eslint/no-unsafe-assignment': 'off',
      '@typescript-eslint/no-unsafe-member-access': 'off',
      '@typescript-eslint/no-unsafe-argument': 'off',
      '@typescript-eslint/no-unsafe-call': 'off',
    },
  },
  // shadcn UI primitives intentionally co-locate component +
  // class-variance-authority variant tables in the same file (it's the
  // canonical shadcn pattern). The router + sidebar likewise export a
  // small constant route table alongside the shell. Fast-refresh's
  // "only export components" rule is a HMR convenience, not a
  // correctness signal — silence it where the co-location is by design.
  {
    files: [
      'src/components/ui/**/*.{ts,tsx}',
      'src/router.tsx',
      'src/components/layout/sidebar.tsx',
    ],
    rules: {
      'react-refresh/only-export-components': 'off',
    },
  },
])
