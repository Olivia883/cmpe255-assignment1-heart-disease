import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Database from "better-sqlite3";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const db = new Database(path.join(__dirname, "tasks.db"));

db.exec(`
  CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    completed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
  )
`);

app.use(express.json());
app.use(express.static(path.join(__dirname, "../client")));

const clients = new Set();

function broadcast() {
  const payload = JSON.stringify({
    type: "tasks",
    tasks: db.prepare("SELECT * FROM tasks ORDER BY id DESC").all()
  });
  for (const res of clients) {
    res.write(`data: ${payload}\n\n`);
  }
}

app.get("/api/tasks", (_req, res) => {
  res.json(db.prepare("SELECT * FROM tasks ORDER BY id DESC").all());
});

app.post("/api/tasks", (req, res) => {
  const title = String(req.body.title || "").trim();
  if (!title) return res.status(400).json({ error: "Title is required." });

  const result = db.prepare(
    "INSERT INTO tasks(title, created_at) VALUES (?, datetime('now'))"
  ).run(title);

  broadcast();
  res.status(201).json({ id: result.lastInsertRowid, title });
});

app.patch("/api/tasks/:id", (req, res) => {
  const id = Number(req.params.id);
  const completed = req.body.completed ? 1 : 0;

  db.prepare("UPDATE tasks SET completed = ? WHERE id = ?")
    .run(completed, id);

  broadcast();
  res.sendStatus(204);
});

app.delete("/api/tasks/:id", (req, res) => {
  db.prepare("DELETE FROM tasks WHERE id = ?").run(Number(req.params.id));
  broadcast();
  res.sendStatus(204);
});

app.get("/events", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.write(": connected\n\n");

  clients.add(res);
  req.on("close", () => clients.delete(res));
});

app.get("*", (_req, res) => {
  res.sendFile(path.join(__dirname, "../client/index.html"));
});

app.listen(5000, () => {
  console.log("Todo Workspace running at http://localhost:5000");
});