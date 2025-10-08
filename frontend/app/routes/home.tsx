import type { Route } from "./+types/home";
import Dashboard from "../welcome/welcome";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Upwork Chatbot" },
    { name: "description", content: "Welcome to React Router!" },
  ];
}

export default function Home() {
  return <Dashboard />;
}
