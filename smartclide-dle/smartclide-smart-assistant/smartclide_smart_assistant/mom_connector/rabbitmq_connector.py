#!/usr/bin/python3
#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************


import json
import pika
import asyncio
import requests
import threading
from time import sleep
from typing import List, Dict, Callable


class RabbitMQConsumer:
    """Consumes from rabbitmq

    Args:
        host: rabbitmq's host
        queues: rabbitmq's queues
        message_handler: handler to receive the message and perform the treatment
    """

    def __init__(self, host:str, port:int, user:str, password:str, queues:List[str] = None, message_handler:Callable = None) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.queues = queues
        self.message_handler = message_handler

    def start(self) -> None:
        
        sleep(5)

        def on_message(ch, method, properties, body):

            try:
                queue = method.routing_key
                print(f'[rabbitmq] message from {queue}')
                body = json.loads(body.decode('utf-8'))
                self.message_handler(queue, body)
            except Exception as e:
                print(f'[rabbitmq] error on message routing: {e}')

        credentials = pika.PlainCredentials(self.user, self.password)
        connection_params = pika.ConnectionParameters(host=self.host, port=self.port, virtual_host='/', credentials=credentials)
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()

        for queue in self.queues:
            print(f'[rabbitmq] subscribing to {queue}')
            channel.queue_declare(queue=queue)
            channel.basic_consume(queue=queue, on_message_callback=on_message, auto_ack=True)

        print('[rabbitmq] connector started')

        channel.start_consuming()


class BackgroundAPIRabbitMQConsumer(RabbitMQConsumer):

    def __init__(self, channel_endpoint_mappings:Dict[str,str], *args, **kwargs):
        queues = list(channel_endpoint_mappings.keys())
        send_to_endpoint = lambda queue, body: requests.post(url=channel_endpoint_mappings[queue], json=body) 
        kwargs['queues'] = queues
        kwargs['message_handler'] = send_to_endpoint
        super().__init__(*args, **kwargs)

    def start(self):    
        t = threading.Thread(target=super().start)
        t.start()
