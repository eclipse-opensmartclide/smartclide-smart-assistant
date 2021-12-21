#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.


import asyncio
import requests
from amqpstorm import Connection
from typing import List, Dict


class RabbitMQConsumer:
    """Consumes from rabbitmq

    Args:
        host: rabbitmq's host
        port: rabbitmq's port
        user: rabbitmq's user
        password: rabbitmq's password
        queue: rabbitmq's queue
        message_handler: handler to receive the message and perform the treatment
    """

    def __init__(self, host:str, user:str, password:str, queues:List[str] = None, message_handler:'function' = None) -> None:
        self.host = host
        self.user = user
        self.queues = queues
        self.password = password
        self.message_handler = message_handler

    def start(self) -> None:
        
        def on_message(message):
            try:
                self.message_handler(message.queue, message.body)
                message.ack()
            except:
                message.reject(requeue=True)

        connection = Connection(self.host, self.user, self.password)
        channel = connection.channel()
        for q in self.queues:
            channel.basic.consume(callback=on_message, queue=q, no_ack=False)
        channel.start_consuming(to_tuple=False)


class BackgroundAPIRabbitMQConsumer(RabbitMQConsumer):

    def __init__(self, channel_endpoint_mappings:Dict[str,str], *args, **kwargs):
        queues = list(channel_endpoint_mappings.values())
        send_to_endpoint = lambda queue, body: requests.post(url=channel_endpoint_mappings[queue], json=body) 
        kwargs['queues'] = queues
        kwargs['message_handler'] = send_to_endpoint
        super().__init__(*args, **kwargs)

    async def _background_start(self):
        super().start()
        
    def start(self):    
        asyncio.run(self._background_start())
