import axios from "axios";

const BASE = process.env.REACT_APP_MCP_URL || "http://localhost:8005/mcp/run";
const TIMEOUT = 30000;

export async function callTool(tool, params = {}) {
  try {
    const res = await axios.post(BASE, { tool, params }, { timeout: TIMEOUT });
    return res.data;
  } catch (err) {
    if (err.response && err.response.data) return err.response.data;
    return { status: "error", message: err.message || "Network error" };
  }
}
