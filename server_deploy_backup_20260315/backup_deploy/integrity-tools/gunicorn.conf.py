import multiprocessing

bind = "0.0.0.0:5006"
workers = 1
worker_class = "sync"
timeout = 300
keepalive = 5
graceful_timeout = 30
errorlog = "-"
accesslog = "-"
loglevel = "info"