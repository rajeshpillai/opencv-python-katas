/* Render root */
import { render } from "solid-js/web";
import { Router } from "@solidjs/router";
import App from "./App";
import "./index.css";
import "./components.css";

const root = document.getElementById("root");
if (!root) throw new Error("No #root element found");

render(() => <Router><App /></Router>, root);
