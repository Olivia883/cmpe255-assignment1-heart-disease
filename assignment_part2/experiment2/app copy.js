const list = document.getElementById("tasks");
const form = document.getElementById("task-form");
const input = document.getElementById("task-title");

async function loadTasks() {
  const response = await fetch("/api/tasks");
  render(await response.json());
}

function render(tasks) {
  list.innerHTML = "";

  let done = 0;
  for (const task of tasks) {
    if (task.completed) done++;

    const li = document.createElement("li");
    li.innerHTML = `
      <label>
        <input type="checkbox" ${task.completed ? "checked" : ""}>
        ${task.title}
      </label>
      <button>Delete</button>
    `;

    li.querySelector("input").addEventListener("change", async (e) => {
      await fetch(`/api/tasks/${task.id}`, {
        method: "PATCH",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ completed: e.target.checked })
      });
    });

    li.querySelector("button").addEventListener("click", async () => {
      await fetch(`/api/tasks/${task.id}`, { method: "DELETE" });
    });

    list.appendChild(li);
  }

  document.getElementById("total").textContent = `Total: ${tasks.length}`;
  document.getElementById("done").textContent = `Done: ${done}`;
  document.getElementById("open").textContent = `Open: ${tasks.length - done}`;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  await fetch("/api/tasks", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ title: input.value })
  });

  input.value = "";
});

const events = new EventSource("/events");
events.onmessage = (event) => {
  const message = JSON.parse(event.data);
  render(message.tasks);
};

loadTasks();
