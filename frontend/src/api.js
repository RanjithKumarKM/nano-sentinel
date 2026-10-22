import axios from "axios";
export const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
export const api = axios.create({ baseURL: API_URL, timeout: 120000 });
export const fileForm = (file) => { const form = new FormData(); form.append("file", file); return form; };
export async function getHealth() { const {data}=await api.get("/health"); return data; }
export async function getHistory() { const {data}=await api.get("/inspections"); return Array.isArray(data) ? data : (data.items || data.inspections || []); }
export async function deleteInspection(id) { return (await api.delete(`/inspections/${encodeURIComponent(id)}`)).data; }
export async function inspectFile(file) { return (await api.post("/inspect", fileForm(file), {headers:{"Content-Type":"multipart/form-data"}})).data; }
export async function explainFile(file) { return (await api.post("/explain", fileForm(file), {headers:{"Content-Type":"multipart/form-data"}})).data; }
export async function explainInspection(id) { return (await api.get(`/inspections/${encodeURIComponent(id)}/explain`)).data; }
export async function askAssistant(question, conversationHistory = []) { return (await api.post("/assistant/ask", {question, conversation_history:conversationHistory})).data; }
export async function getReport(id) { return (await api.get(`/inspections/${encodeURIComponent(id)}/report`, {responseType:"blob"})).data; }
export function saveBlob(blob, filename) { const url=URL.createObjectURL(blob); const a=document.createElement("a"); a.href=url; a.download=filename; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); }
export function errorMessage(err) { return err?.response?.data?.detail || err?.message || "Unexpected error"; }