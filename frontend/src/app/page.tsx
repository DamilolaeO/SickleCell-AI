import { Chat } from "@/components/chat";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col px-4">
      <header className="border-b py-4">
        <h1 className="text-2xl font-semibold">Sicklewise</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Trusted information about sickle cell disease from the NHS and other reliable sources.
          This app cannot check your symptoms or tell you what medicine to take. In an emergency,
          call 999.
        </p>
      </header>
      <Chat />
    </main>
  );
}
