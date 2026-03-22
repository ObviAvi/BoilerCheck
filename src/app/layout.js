import "./globals.css";

export const metadata = {
  title: "BoilerCheck",
  description: "Purdue policy answers with sources",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}