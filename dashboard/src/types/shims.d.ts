// cytoscape-fcose ships no type declarations.
declare module "cytoscape-fcose" {
  import type { Ext } from "cytoscape";
  const ext: Ext;
  export default ext;
}

// react-cytoscapejs ships no type declarations — minimal surface used here.
declare module "react-cytoscapejs" {
  import type { CSSProperties } from "react";
  import type {
    Core,
    ElementDefinition,
    LayoutOptions,
    StylesheetJson,
  } from "cytoscape";

  export interface CytoscapeComponentProps {
    elements: ElementDefinition[];
    style?: CSSProperties;
    className?: string;
    stylesheet?: StylesheetJson;
    layout?: LayoutOptions;
    cy?: (cy: Core) => void;
    minZoom?: number;
    maxZoom?: number;
    wheelSensitivity?: number;
  }

  const CytoscapeComponent: (props: CytoscapeComponentProps) => JSX.Element;
  export default CytoscapeComponent;
}
