import sys
import collections
from enum import Enum
import time

import colorama
from mpi4py import MPI


comm = MPI.COMM_WORLD


class LamportClock:
    value = 0

    @staticmethod
    def cmp(x):
        if x > LamportClock.value:
            LamportClock.value = x
        LamportClock.value += 1

    @staticmethod
    def inc():
        LamportClock.value += 1
        return LamportClock.value


class NodeState(Enum):
    NO_TOKEN = 0
    TOKEN_PING = 1
    TOKEN_PONG = 2
    BOTH_TOKENS = 3


class MessageTag(Enum):
    TOKEN = 0


class TokenType(Enum):
    PING = 0
    PONG = 1


class Logger:

    colors = {
        'info': colorama.Fore.GREEN,
        'warning': colorama.Fore.YELLOW,
        'critical': colorama.Fore.RED
    }

    def __init__(self):
        self.rank = comm.rank

    def _message(self, color, message):
        print(colorama.Fore.MAGENTA + "%05d\t" % (LamportClock.value) +
              colorama.Fore.CYAN + ("%02d\t" % self.rank) +
              color + message, file=sys.stderr, flush=True)

    def info(self, message):
        self._message(Logger.colors['info'], message)

    def warning(self, message):
        self._message(Logger.colors['warning'], message)

    def critical(self, message):
        self._message(Logger.colors['critical'], message)


class Message:
    def __init__(self, message, tag):
        self.message = message
        self.tag = tag.value
        self.lamport = LamportClock.inc()

    def send(self, recipients):
        if isinstance(recipients, collections.Iterable):
            for i in recipients:
                comm.send((self.lamport, self.message), i, self.tag)
        else:
            comm.send((self.lamport, self.message), recipients, self.tag)

    def broadcast(self):
        for node in range(comm.size):
            if node != comm.rank:
                comm.send((self.lamport, self.message), node, self.tag)

    @staticmethod
    def handle(message):
        LamportClock.cmp(message[0])
        return message[1]


class Token:

    def __init__(self, token_type, value):
        if not isinstance(token_type, TokenType):
            raise RuntimeError("Token should be TokenType instance")
        self.type = token_type
        self.value = value

    def __str__(self):
        return "TOKEN %s(%d)" % (self.type, self.value)

    def send(self, recipient):
        Logger().info("Sending %s to node %d" % (self, recipient))
        Message(self, MessageTag.TOKEN).send(recipient)


class Node:

    def __init__(self, comm):
        self.token_ping = None
        self.token_pong = None
        self.rank = comm.rank
        self.next_node = (comm.rank + 1) % comm.size
        self.m = 0
        self.state = NodeState.NO_TOKEN
        self.log = Logger()
        self.log.info("initialized node")

    def run(self):
        self.log.info("starting node")
        if self.rank == 0:
            self.log.warning("I am the chosen one, generating first tokens")
            Token(TokenType.PING, 1).send(self.next_node)
            Token(TokenType.PONG, -1).send(self.next_node)
            self.m = -1

        while True:
            self.log.info('state %s' % self.state.name)

            if self.state == NodeState.NO_TOKEN:
                self.listen()

            elif self.state == NodeState.TOKEN_PING:
                self.log.info("entering Critical Section")
                self.log.info("leaving Critical Section")

                self.m = self.token_ping.value
                self.token_ping.send(self.next_node)
                self.token_ping = None
                self.state = NodeState.NO_TOKEN

            elif self.state == NodeState.TOKEN_PONG:
                self.m = self.token_pong.value
                self.token_pong.send(self.next_node)
                self.token_pong = None
                self.state = NodeState.NO_TOKEN

            elif self.state == NodeState.BOTH_TOKENS:
                self.incarnate_tokens(self.token_ping.value)

    def listen(self):
        token = Message.handle(comm.recv())
        self.log.info("received token %s" % token)
        self.become(token)

    def become(self, token):
        # let the ifs begin
        if token.value == self.m:
            if self.m > 0:
                self.log.warning('PONG TOKEN LOST')
                self.regenerate_token(TokenType.PONG, token)
            else:
                self.log.warning('PING TOKEN LOST')
                self.regenerate_token(TokenType.PING, token)
        elif abs(token.value) < abs(self.m):
            self.log.critical("received some old token %s, deleting" % token)
            return None

        if token.type == TokenType.PING:
            self.token_ping = token
            if self.state == NodeState.NO_TOKEN:
                self.state = NodeState.TOKEN_PING

            elif self.state == NodeState.TOKEN_PONG:
                self.state = NodeState.BOTH_TOKENS

            elif self.state in [NodeState.TOKEN_PING, NodeState.BOTH_TOKENS]:
                self.log.critical("ok, now we can panic")

        if token.type == TokenType.PONG:
            self.token_pong = token
            if self.state == NodeState.NO_TOKEN:
                self.state = NodeState.TOKEN_PONG
            elif self.state == NodeState.TOKEN_PING:
                self.state = NodeState.BOTH_TOKENS
            elif self.state in [NodeState.TOKEN_PONG, NodeState.BOTH_TOKENS]:
                self.log.critical("ok, now we can panic")

    def incarnate_tokens(self, x):
        return Token(TokenType.PING, abs(x)),\
            Token(TokenType.PONG, - abs(x + 1))

    def regenerate_token(self, x):
        pass


if __name__ == '__main__':
    Node(comm).run()
