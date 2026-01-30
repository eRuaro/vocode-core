import hashlib
from unittest.mock import MagicMock, patch

import pytest
from pydantic.v1 import ValidationError
from pytest_mock import MockerFixture

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.synthesizer import CambAiSynthesizerConfig
from vocode.streaming.synthesizer.camb_ai_synthesizer import CambAiSynthesizer
from vocode.streaming.synthesizer.default_factory import DefaultSynthesizerFactory

DEFAULT_PARAMS = {
    "sampling_rate": 22050,  # mars-flash native sample rate
    "audio_encoding": AudioEncoding.LINEAR16,
}


@pytest.fixture(autouse=True)
def mock_async_requestor(mocker: MockerFixture):
    """Mock AsyncRequestor to avoid needing an event loop for non-async tests."""
    mocker.patch(
        "vocode.streaming.synthesizer.base_synthesizer.AsyncRequestor",
        return_value=MagicMock(),
    )


def test_camb_ai_synthesizer_config_defaults():
    config = CambAiSynthesizerConfig(**DEFAULT_PARAMS)
    assert config.voice_id == 147320  # DEFAULT_CAMB_AI_VOICE_ID
    assert config.model_id == "mars-flash"
    assert config.language == "en-us"  # DEFAULT_CAMB_AI_LANGUAGE
    assert config.user_instructions is None
    assert config.enhance_named_entities_pronunciation is False


def test_camb_ai_synthesizer_config_custom_values():
    config = CambAiSynthesizerConfig(
        **DEFAULT_PARAMS,
        api_key="test_key",
        voice_id=1234,
        model_id="mars-pro",
        language="fr-fr",
        enhance_named_entities_pronunciation=True,
    )
    assert config.api_key == "test_key"
    assert config.voice_id == 1234
    assert config.model_id == "mars-pro"
    assert config.language == "fr-fr"
    assert config.enhance_named_entities_pronunciation is True


def test_camb_ai_synthesizer_config_user_instructions_requires_instruct_model():
    with pytest.raises(ValidationError) as exc_info:
        CambAiSynthesizerConfig(
            **DEFAULT_PARAMS,
            model_id="mars-flash",
            user_instructions="Speak slowly and clearly.",
        )
    assert "user_instructions is only supported with mars-instruct model" in str(exc_info.value)


def test_camb_ai_synthesizer_config_user_instructions_with_instruct_model():
    config = CambAiSynthesizerConfig(
        **DEFAULT_PARAMS,
        model_id="mars-instruct",
        user_instructions="Speak slowly and clearly.",
    )
    assert config.user_instructions == "Speak slowly and clearly."


def test_camb_ai_synthesizer_config_user_instructions_length_validation():
    with pytest.raises(ValidationError) as exc_info:
        CambAiSynthesizerConfig(
            **DEFAULT_PARAMS,
            model_id="mars-instruct",
            user_instructions="ab",
        )
    assert "must be between 3 and 1000 characters" in str(exc_info.value)


def test_camb_ai_synthesizer_requires_api_key():
    config = CambAiSynthesizerConfig(**DEFAULT_PARAMS)
    with patch.dict("os.environ", {}, clear=True):
        with patch("vocode.getenv", return_value=None):
            with pytest.raises(ValueError) as exc_info:
                CambAiSynthesizer(config)
            assert "Camb.ai API key is required" in str(exc_info.value)


def test_camb_ai_synthesizer_uses_config_api_key():
    config = CambAiSynthesizerConfig(**DEFAULT_PARAMS, api_key="config_api_key")
    with patch("vocode.getenv", return_value=None):
        synthesizer = CambAiSynthesizer(config)
        assert synthesizer.api_key == "config_api_key"


def test_camb_ai_synthesizer_uses_env_api_key():
    config = CambAiSynthesizerConfig(**DEFAULT_PARAMS)
    with patch(
        "vocode.streaming.synthesizer.camb_ai_synthesizer.getenv", return_value="env_api_key"
    ):
        synthesizer = CambAiSynthesizer(config)
        assert synthesizer.api_key == "env_api_key"


def test_camb_ai_synthesizer_voice_identifier():
    config = CambAiSynthesizerConfig(
        **DEFAULT_PARAMS,
        api_key="test_key",
        voice_id=147320,
        model_id="mars-flash",
        language="en-us",
    )
    voice_id = CambAiSynthesizer.get_voice_identifier(config)

    hashed_api_key = hashlib.sha256("test_key".encode("utf-8")).hexdigest()
    expected = f"camb_ai:{hashed_api_key}:147320:mars-flash:en-us:False:linear16"
    assert voice_id == expected


def test_camb_ai_synthesizer_needs_resample_for_non_native_rate():
    # mars-flash native rate is 22050, so 16000 needs resampling
    config = CambAiSynthesizerConfig(
        sampling_rate=16000,
        audio_encoding=AudioEncoding.LINEAR16,
        api_key="test_key",
        model_id="mars-flash",
    )
    synthesizer = CambAiSynthesizer(config)
    assert synthesizer.needs_resample is True
    assert synthesizer.target_sample_rate == 16000
    assert synthesizer.native_sample_rate == 22050


def test_camb_ai_synthesizer_no_resample_for_native_rate():
    # mars-flash native rate is 22050
    config = CambAiSynthesizerConfig(
        sampling_rate=22050,
        audio_encoding=AudioEncoding.LINEAR16,
        api_key="test_key",
        model_id="mars-flash",
    )
    synthesizer = CambAiSynthesizer(config)
    assert synthesizer.needs_resample is False
    assert synthesizer.native_sample_rate == 22050


def test_camb_ai_synthesizer_mars_pro_native_rate():
    # mars-pro native rate is 48000
    config = CambAiSynthesizerConfig(
        sampling_rate=48000,
        audio_encoding=AudioEncoding.LINEAR16,
        api_key="test_key",
        model_id="mars-pro",
    )
    synthesizer = CambAiSynthesizer(config)
    assert synthesizer.needs_resample is False
    assert synthesizer.native_sample_rate == 48000


def test_camb_ai_synthesizer_mulaw_needs_resample():
    config = CambAiSynthesizerConfig(
        sampling_rate=8000,
        audio_encoding=AudioEncoding.MULAW,
        api_key="test_key",
    )
    synthesizer = CambAiSynthesizer(config)
    assert synthesizer.needs_resample is True


def test_factory_creates_camb_ai_synthesizer(mocker: MockerFixture):
    factory = DefaultSynthesizerFactory()

    constructor = mocker.patch("vocode.streaming.synthesizer.default_factory.CambAiSynthesizer")

    factory.create_synthesizer(CambAiSynthesizerConfig(**DEFAULT_PARAMS, api_key="test_key"))
    constructor.assert_called_once()
