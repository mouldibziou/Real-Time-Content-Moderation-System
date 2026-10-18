import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource-variable/instrument-sans";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import { Landing } from "./Landing";
import { Demo } from "./Demo";
import "./styles.css";
const isDemo = window.location.pathname.replace(/\/$/, "") === "/demo";
if (isDemo) document.title = "Try Flagged — The moderation workspace";
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <a className="skip-link" href="#main-content">
      Skip to content
    </a>
    {isDemo ? <Demo /> : <Landing />}
  </React.StrictMode>,
);
