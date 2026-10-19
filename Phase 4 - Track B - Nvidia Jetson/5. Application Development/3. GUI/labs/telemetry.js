/* One-clock teaching model. No network or actuator control. */
(function (root) {
    'use strict';
    const finite = value => typeof value === 'number' && Number.isFinite(value);
    function create(session) {
        if (typeof session !== 'string' || session.length < 1 || session.length > 64)
            throw new Error('session must be a nonempty bounded string');
        return {session, connected: true, sample: null, invalid: '', rejected: 0};
    }
    function receive(state, message, now) {
        if (!state.connected) return state;
        let reason = '';
        if (!finite(now) || now < 0) reason = 'Invalid local clock';
        else if (!message || typeof message !== 'object' || Array.isArray(message)) reason = 'Invalid message';
        else if (message.session !== state.session) reason = 'Wrong session';
        else if (!Number.isSafeInteger(message.sequence) || message.sequence < 0) reason = 'Invalid sequence';
        else if (!finite(message.captured_ms) || message.captured_ms < 0 || message.captured_ms > now)
            reason = 'Invalid capture time';
        else if (!finite(message.distance_m) || message.distance_m < 0 || message.distance_m > 20)
            reason = 'Invalid distance';
        else if (state.sample && (message.sequence <= state.sample.sequence || message.captured_ms < state.sample.captured_ms || now < state.sample.received_ms))
            reason = 'Duplicate, old sample or clock reversal';
        if (reason) return {...state, invalid: reason, rejected: state.rejected + 1};
        const sample = Object.freeze({session: message.session, sequence: message.sequence,
            captured_ms: message.captured_ms, received_ms: now, distance_m: message.distance_m});
        return {...state, sample, invalid: ''};
    }
    function disconnect(state) { return {...state, connected: false}; }
    function view(state, now, deadlineMs = 250) {
        if (!finite(now) || now < 0 || !finite(deadlineMs) || deadlineMs <= 0)
            throw new Error('finite clock and positive deadline required');
        const age = state.sample ? now - state.sample.captured_ms : null;
        if (!state.connected) return {status: 'Disconnected', tone: 'bad', age_ms: age, current: false};
        if (state.invalid) return {status: 'Invalid sample', tone: 'bad', age_ms: age, current: false};
        if (!state.sample) return {status: 'Waiting for data', tone: 'waiting', age_ms: null, current: false};
        if (now < state.sample.received_ms) return {status: 'Clock error', tone: 'bad', age_ms: age, current: false};
        if (age > deadlineMs) return {status: 'Stale observation', tone: 'waiting', age_ms: age, current: false};
        return {status: 'Fresh observation', tone: 'good', age_ms: age, current: true};
    }
    const api = {create, receive, disconnect, view};
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.CourseTelemetry = api;
})(globalThis);
