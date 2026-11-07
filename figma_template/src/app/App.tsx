import { FormEvent, useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import {
  Bot,
  Check,
  ChevronDown,
  FileText,
  Menu,
  MessageSquareText,
  MoreHorizontal,
  Paperclip,
  PanelLeftClose,
  Plus,
  Search,
  Send,
  Settings,
  Sparkles,
  Trash2,
  UserRound,
  X,
} from "lucide-react";

type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
  source?: string;
};

const suggestions = [
  {
    title: "Tóm tắt nội dung chính",
    detail: "của toàn bộ tài liệu đã tải lên",
  },
  {
    title: "Giải thích Nested Learning",
    detail: "bằng một ví dụ dễ hiểu",
  },
  {
    title: "So sánh các phương pháp",
    detail: "được đề cập trong tài liệu",
  },
  {
    title: "Tạo 5 câu hỏi ôn tập",
    detail: "kèm đáp án và nguồn trích dẫn",
  },
];

const initialMessages: Message[] = [];

function Logo() {
  return (
    <div className="grid size-8 shrink-0 place-items-center rounded-lg bg-[#7257FF] text-white">
      <Sparkles size={17} strokeWidth={2.5} />
    </div>
  );
}

function Sidebar({
  open,
  onClose,
  onNewChat,
}: {
  open: boolean;
  onClose: () => void;
  onNewChat: () => void;
}) {
  return (
    <>
      {open && (
        <button
          aria-label="Đóng thanh bên"
          className="fixed inset-0 z-30 bg-black/25 lg:hidden"
          onClick={onClose}
        />
      )}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[280px] flex-col border-r border-[#E5E7EB] bg-[#F8F8FA] transition-transform duration-300 lg:static lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between px-3 py-3">
          <button
            onClick={onNewChat}
            className="flex flex-1 items-center gap-3 rounded-xl p-2 text-left transition hover:bg-[#ECECF1]"
          >
            <Logo />
            <span className="font-display text-[15px] font-bold">NL DocChat</span>
          </button>
          <button
            onClick={onClose}
            className="grid size-9 place-items-center rounded-lg text-[#6B7280] transition hover:bg-[#ECECF1] hover:text-[#171A24]"
            aria-label="Thu gọn thanh bên"
          >
            <PanelLeftClose size={19} />
          </button>
        </div>

        <div className="px-3">
          <button
            onClick={onNewChat}
            className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-semibold transition hover:bg-[#ECECF1]"
          >
            <Plus size={18} />
            Cuộc trò chuyện mới
          </button>
          <button className="mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition hover:bg-[#ECECF1]">
            <Search size={18} />
            Tìm kiếm cuộc trò chuyện
          </button>
        </div>

        <div className="mt-6 flex-1 overflow-y-auto px-3">
          <p className="px-3 pb-2 text-xs font-semibold text-[#8B8F98]">Gần đây</p>
          {[
            "Nested Learning là gì?",
            "Tóm tắt báo cáo nghiên cứu",
            "So sánh RAG và Fine-tuning",
            "Các kết quả thử nghiệm",
          ].map((chat, index) => (
            <button
              key={chat}
              className={`group mb-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm transition ${
                index === 0 ? "bg-[#ECECF1] font-medium" : "hover:bg-[#ECECF1]"
              }`}
            >
              <MessageSquareText size={16} className="shrink-0 text-[#777C87]" />
              <span className="min-w-0 flex-1 truncate">{chat}</span>
              <MoreHorizontal
                size={16}
                className="shrink-0 opacity-0 transition group-hover:opacity-100"
              />
            </button>
          ))}

          <p className="mt-6 px-3 pb-2 text-xs font-semibold text-[#8B8F98]">Tài liệu</p>
          <button className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm transition hover:bg-[#ECECF1]">
            <span className="grid size-7 shrink-0 place-items-center rounded-md bg-[#E9E5FF] text-[#7257FF]">
              <FileText size={15} />
            </span>
            <span className="min-w-0 flex-1 truncate">nested-learning.pdf</span>
            <Check size={15} className="text-[#00A987]" />
          </button>
          <button className="mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm text-[#626773] transition hover:bg-[#ECECF1]">
            <span className="grid size-7 place-items-center rounded-md border border-dashed border-[#AEB2BB]">
              <Plus size={14} />
            </span>
            Thêm tài liệu
          </button>
        </div>

        <div className="border-t border-[#E5E7EB] p-3">
          <button className="flex w-full items-center gap-3 rounded-xl p-2 text-left transition hover:bg-[#ECECF1]">
            <div className="grid size-9 shrink-0 place-items-center rounded-full bg-[#F7CF45] text-sm font-bold">
              NA
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-semibold">Nguyễn An</p>
              <p className="text-xs text-[#8B8F98]">Gói miễn phí</p>
            </div>
            <Settings size={17} className="text-[#777C87]" />
          </button>
        </div>
      </aside>
    </>
  );
}

function Composer({
  value,
  onChange,
  onSubmit,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
}) {
  return (
    <form
      onSubmit={(event: FormEvent) => {
        event.preventDefault();
        onSubmit();
      }}
      className="mx-auto w-full max-w-3xl px-3 pb-3 sm:px-5 sm:pb-5"
    >
      <div className="rounded-[26px] border border-[#D8DAE0] bg-white p-2 shadow-[0_8px_30px_rgba(23,26,36,0.10)] transition focus-within:border-[#AEB2BB]">
        <textarea
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onSubmit();
            }
          }}
          rows={1}
          placeholder="Hỏi bất kỳ điều gì về tài liệu..."
          className="block max-h-40 min-h-[48px] w-full resize-none bg-transparent px-3 py-3 text-[15px] leading-6 outline-none placeholder:text-[#9296A0]"
        />
        <div className="flex items-center justify-between px-1 pb-1">
          <div className="flex items-center gap-1">
            <button
              type="button"
              className="grid size-9 place-items-center rounded-full text-[#555B66] transition hover:bg-[#F0F1F4]"
              aria-label="Đính kèm tài liệu"
            >
              <Paperclip size={19} />
            </button>
            <button
              type="button"
              className="flex h-9 items-center gap-2 rounded-full px-3 text-xs font-semibold text-[#555B66] transition hover:bg-[#F0F1F4]"
            >
              <FileText size={16} />
              <span className="hidden sm:inline">1 tài liệu</span>
            </button>
          </div>
          <button
            type="submit"
            disabled={!value.trim()}
            className="grid size-9 place-items-center rounded-full bg-[#171A24] text-white transition hover:bg-[#7257FF] disabled:bg-[#D5D7DC] disabled:text-[#8D9199]"
            aria-label="Gửi câu hỏi"
          >
            <Send size={17} />
          </button>
        </div>
      </div>
      <p className="mt-2 text-center text-[11px] text-[#8B8F98]">
        NL DocChat có thể mắc lỗi. Hãy kiểm tra thông tin quan trọng từ nguồn trích dẫn.
      </p>
    </form>
  );
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [thinking, setThinking] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  const sendMessage = (preset?: string) => {
    const content = (preset ?? input).trim();
    if (!content || thinking) return;

    setMessages((current) => [
      ...current,
      { id: Date.now(), role: "user", content },
    ]);
    setInput("");
    setThinking(true);

    window.setTimeout(() => {
      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: "assistant",
          content:
            "Nested Learning là phương pháp tổ chức quá trình học thành nhiều tầng lồng nhau. Ở mỗi tầng, hệ thống xử lý một mức độ ngữ cảnh khác nhau — từ ý định tổng quát, các đoạn liên quan cho đến bằng chứng chi tiết. Nhờ vậy, chatbot không chỉ tìm từ khóa mà còn hiểu mối liên hệ giữa các phần trong tài liệu, giúp câu trả lời chính xác và nhất quán hơn.",
          source: "nested-learning.pdf · Trang 8–9",
        },
      ]);
      setThinking(false);
    }, 900);
  };

  const newChat = () => {
    setMessages([]);
    setInput("");
    setSidebarOpen(false);
  };

  return (
    <main className="flex h-screen overflow-hidden bg-white font-sans text-[#171A24]">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={newChat}
      />

      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-[#ECEDEF] bg-white px-3 sm:px-5">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setSidebarOpen(true)}
              className="grid size-9 place-items-center rounded-lg transition hover:bg-[#F0F1F4] lg:hidden"
              aria-label="Mở thanh bên"
            >
              <Menu size={20} />
            </button>
            <button className="flex items-center gap-2 rounded-lg px-2 py-1.5 font-display text-[15px] font-bold transition hover:bg-[#F5F5F7]">
              NL DocChat
              <span className="hidden font-sans text-xs font-medium text-[#777C87] sm:inline">
                Nested Learning
              </span>
              <ChevronDown size={15} className="text-[#777C87]" />
            </button>
          </div>
          <div className="flex items-center gap-2">
            {messages.length > 0 && (
              <button
                onClick={newChat}
                className="hidden items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition hover:bg-[#F0F1F4] sm:flex"
              >
                <Trash2 size={16} />
                Xóa cuộc trò chuyện
              </button>
            )}
            <button className="flex items-center gap-2 rounded-full border border-[#E0E2E6] px-3 py-1.5 text-xs font-semibold transition hover:bg-[#F8F8FA]">
              <span className="size-2 rounded-full bg-[#28CFA8]" />
              <span className="hidden sm:inline">Tài liệu đã sẵn sàng</span>
              <span className="sm:hidden">Sẵn sàng</span>
            </button>
          </div>
        </header>

        {messages.length === 0 ? (
          <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
            <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center px-5 py-10">
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45 }}
              >
                <div className="mb-6 flex items-center justify-center">
                  <div className="relative">
                    <div className="grid size-14 place-items-center rounded-2xl bg-[#7257FF] text-white shadow-[0_8px_24px_rgba(114,87,255,0.25)]">
                      <Bot size={27} />
                    </div>
                    <span className="absolute -bottom-1 -right-1 size-4 rounded-full border-2 border-white bg-[#28CFA8]" />
                  </div>
                </div>
                <h1 className="text-center font-display text-[clamp(1.9rem,4vw,2.6rem)] font-bold leading-tight tracking-[-0.035em]">
                  Hôm nay tôi có thể giúp gì cho bạn?
                </h1>
                <p className="mx-auto mt-3 max-w-xl text-center text-sm leading-6 text-[#6B707B]">
                  Hỏi đáp, tóm tắt và khám phá kiến thức từ tài liệu của bạn bằng Nested Learning.
                </p>

                <div className="mt-9 grid gap-2 sm:grid-cols-2">
                  {suggestions.map((suggestion, index) => (
                    <motion.button
                      key={suggestion.title}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.08 * index }}
                      onClick={() =>
                        sendMessage(`${suggestion.title} ${suggestion.detail}`)
                      }
                      className="group rounded-2xl border border-[#E1E3E7] p-4 text-left transition hover:border-[#C9C4F8] hover:bg-[#F8F7FF]"
                    >
                      <p className="text-sm font-semibold group-hover:text-[#6247E6]">
                        {suggestion.title}
                      </p>
                      <p className="mt-1 text-xs leading-5 text-[#7B808B]">
                        {suggestion.detail}
                      </p>
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            </div>
            <Composer
              value={input}
              onChange={setInput}
              onSubmit={() => sendMessage()}
            />
          </div>
        ) : (
          <>
            <div className="min-h-0 flex-1 overflow-y-auto">
              <div className="mx-auto w-full max-w-3xl px-5 py-8 sm:py-12">
                {messages.map((message) => (
                  <motion.article
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    key={message.id}
                    className="mb-9 grid grid-cols-[32px_minmax(0,1fr)] gap-4 sm:grid-cols-[36px_minmax(0,1fr)] sm:gap-5"
                  >
                    <div
                      className={`grid size-8 place-items-center rounded-full sm:size-9 ${
                        message.role === "assistant"
                          ? "bg-[#7257FF] text-white"
                          : "bg-[#F7CF45] text-[#171A24]"
                      }`}
                    >
                      {message.role === "assistant" ? (
                        <Sparkles size={17} />
                      ) : (
                        <UserRound size={17} />
                      )}
                    </div>
                    <div className="pt-1">
                      <p className="mb-2 text-sm font-bold">
                        {message.role === "assistant" ? "NL DocChat" : "Bạn"}
                      </p>
                      <p className="whitespace-pre-wrap text-[15px] leading-7 text-[#343842]">
                        {message.content}
                      </p>
                      {message.source && (
                        <button className="mt-4 flex items-center gap-2 rounded-lg border border-[#E0E2E7] bg-[#FAFAFB] px-3 py-2 text-xs font-medium text-[#5F6470] transition hover:border-[#B9B2FA] hover:bg-[#F6F4FF]">
                          <FileText size={15} className="text-[#7257FF]" />
                          {message.source}
                        </button>
                      )}
                    </div>
                  </motion.article>
                ))}

                {thinking && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mb-9 grid grid-cols-[32px_minmax(0,1fr)] gap-4 sm:grid-cols-[36px_minmax(0,1fr)] sm:gap-5"
                  >
                    <div className="grid size-8 place-items-center rounded-full bg-[#7257FF] text-white sm:size-9">
                      <Sparkles size={17} />
                    </div>
                    <div className="pt-1">
                      <p className="mb-3 text-sm font-bold">NL DocChat</p>
                      <div className="flex h-5 items-center gap-1">
                        {[0, 1, 2].map((item) => (
                          <motion.span
                            key={item}
                            animate={{ opacity: [0.25, 1, 0.25], y: [0, -3, 0] }}
                            transition={{
                              duration: 1,
                              repeat: Infinity,
                              delay: item * 0.16,
                            }}
                            className="size-1.5 rounded-full bg-[#777C87]"
                          />
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
                <div ref={bottomRef} />
              </div>
            </div>
            <div className="shrink-0 bg-white/95 pt-2 backdrop-blur">
              <Composer
                value={input}
                onChange={setInput}
                onSubmit={() => sendMessage()}
              />
            </div>
          </>
        )}
      </section>
    </main>
  );
}
