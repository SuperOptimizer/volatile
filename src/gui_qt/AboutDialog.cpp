#include "AboutDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QDesktopServices>
#include <QUrl>
#include <QIcon>
#include <QApplication>
#include <QStyle>

static constexpr const char *k_version   = "0.1.0";
static constexpr const char *k_repo_url  = "https://github.com/ScrollPrize/volatile";
static constexpr const char *k_license   =
  "MIT License\n\n"
  "Copyright (c) 2025 Volatile Contributors\n\n"
  "Permission is hereby granted, free of charge, to any person obtaining a copy "
  "of this software and associated documentation files (the \"Software\"), to deal "
  "in the Software without restriction, including without limitation the rights "
  "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell "
  "copies of the Software, and to permit persons to whom the Software is "
  "furnished to do so, subject to the following conditions:\n\n"
  "The above copyright notice and this permission notice shall be included in all "
  "copies or substantial portions of the Software.\n\n"
  "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR "
  "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, "
  "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE "
  "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER "
  "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, "
  "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE "
  "SOFTWARE.";

AboutDialog::AboutDialog(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("About Volatile"));
  setMinimumWidth(480);
  resize(480, 400);

  auto *root = new QVBoxLayout(this);
  root->setSpacing(12);
  root->setContentsMargins(20, 20, 20, 20);

  // --- Header: icon + title ---
  auto *header = new QHBoxLayout;
  auto *icon_label = new QLabel;
  QIcon app_icon = QApplication::windowIcon();
  if (app_icon.isNull())
    app_icon = style()->standardIcon(QStyle::SP_ComputerIcon);
  icon_label->setPixmap(app_icon.pixmap(48, 48));
  header->addWidget(icon_label);

  auto *title_col = new QVBoxLayout;
  auto *title = new QLabel(QStringLiteral("<b>Volatile v%1</b>").arg(k_version));
  title->setTextFormat(Qt::RichText);
  auto *subtitle = new QLabel(tr("Scientific volume viewer for the Vesuvius Challenge"));
  subtitle->setWordWrap(true);
  title_col->addWidget(title);
  title_col->addWidget(subtitle);
  header->addLayout(title_col, 1);
  root->addLayout(header);

  // --- Build info ---
  auto *build_label = new QLabel(
    QStringLiteral("Qt %1  |  Build: %2 %3")
      .arg(QLatin1StringView{QT_VERSION_STR}, QLatin1StringView{__DATE__},
           QLatin1StringView{__TIME__}));
  build_label->setStyleSheet(QStringLiteral("color: gray; font-size: 11px;"));
  root->addWidget(build_label);

  // --- GitHub link ---
  auto *link = new QLabel(
    QStringLiteral("<a href=\"%1\">%1</a>").arg(QLatin1StringView{k_repo_url}));
  link->setTextFormat(Qt::RichText);
  link->setTextInteractionFlags(Qt::TextBrowserInteraction);
  link->setOpenExternalLinks(false);
  connect(link, &QLabel::linkActivated, this, [](const QString &url) {
    QDesktopServices::openUrl(QUrl(url));
  });
  root->addWidget(link);

  // --- License (scrollable) ---
  auto *lic_label = new QLabel(tr("License:"));
  root->addWidget(lic_label);
  auto *lic_text = new QTextEdit;
  lic_text->setReadOnly(true);
  lic_text->setPlainText(QLatin1StringView{k_license});
  lic_text->setFixedHeight(150);
  root->addWidget(lic_text);

  // --- OK button ---
  auto *btn_row = new QHBoxLayout;
  btn_row->addStretch();
  auto *ok = new QPushButton(tr("OK"));
  ok->setDefault(true);
  connect(ok, &QPushButton::clicked, this, &QDialog::accept);
  btn_row->addWidget(ok);
  root->addLayout(btn_row);
}
