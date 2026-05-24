import { useEffect, useState } from "react";

export type Theme = "dark" | "light";

/** Subscribe to the canonical dark/light state on `<html>`.
 *
 * The class is mutated by the pre-paint bootstrap (index.html) and by
 * `ThemeToggle`, so observing the attribute is the single source of
 * truth — no extra event bus or shared context needed. */
export function useDocumentTheme(): Theme {
  const [theme, setTheme] = useState<Theme>(() =>
    typeof document !== "undefined" &&
    document.documentElement.classList.contains("dark")
      ? "dark"
      : "light",
  );
  useEffect(() => {
    const root = document.documentElement;
    const update = () =>
      setTheme(root.classList.contains("dark") ? "dark" : "light");
    update();
    const obs = new MutationObserver(update);
    obs.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return theme;
}
