#include <QApplication>
#include "MainWindow.h"

extern "C" {
#include "core/log.h"
}

int main(int argc, char *argv[]) {
  QApplication app(argc, argv);
  app.setApplicationName("Volatile");
  app.setOrganizationName("SuperOptimizer");
  app.setApplicationVersion("0.1.0");

  log_set_level(LOG_INFO);
  LOG_INFO("Volatile Qt GUI starting");

  MainWindow w;
  w.show();

  if (argc > 1) w.openVolume(argv[1]);

  return app.exec();
}
